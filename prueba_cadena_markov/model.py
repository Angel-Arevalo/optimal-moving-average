"""Gaussian HMM training and explicit log-space causal forward filtering."""
from dataclasses import asdict
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
from threadpoolctl import threadpool_limits

from .config import FEATURES, MarkovConfig


class MarkovRegimeFilter:
    def __init__(self, config=MarkovConfig()):
        self.config = config
        self.feature_order = FEATURES
        self.valid = False
        self.reason = "not_fitted"
        self.favorable_states = set()
        self.scaler = StandardScaler()
        self.model = None
        self.fit_start = self.fit_end = None
        self.calibration_end = None
        self.diagnostics = {}

    def _check_frame(self, features):
        if tuple(features.columns) != self.feature_order:
            raise ValueError("Feature order differs from fitted schema")
        if (not isinstance(features.index, pd.DatetimeIndex)
                or not features.index.is_monotonic_increasing
                or features.index.has_duplicates or features.index.hasnans):
            raise ValueError("Features require unique ordered datetimes")

    def fit(self, features, *, available_by):
        self._check_frame(features)
        self.valid, self.reason = False, "insufficient_observations"
        self.favorable_states = set()
        self.calibration_end = None
        self.model = None
        self.fit_start = self.fit_end = None
        self.scaler = StandardScaler()
        self.diagnostics = {}
        if len(features) and features.index[-1] > pd.Timestamp(available_by):
            raise ValueError("Fit observations extend beyond the historical cutoff")
        good = np.isfinite(features.to_numpy()).all(axis=1)
        clean = features.loc[good]
        if len(clean) < self.config.min_observations:
            return self
        self.fit_start, self.fit_end = clean.index[0], clean.index[-1]
        # Break the EM sequences at missing rows rather than linking over gaps.
        positions = np.flatnonzero(good)
        lengths = np.diff(np.r_[0, np.flatnonzero(np.diff(positions) > 1) + 1, len(positions)])
        try:
            scaled = self.scaler.fit_transform(clean)
            self.model = GaussianHMM(
                n_components=self.config.n_components, covariance_type="diag",
                n_iter=self.config.n_iter, tol=self.config.tol,
                random_state=self.config.random_state, min_covar=self.config.covariance_floor,
                implementation="log")
            with threadpool_limits(limits=1):
                self.model.fit(scaled, lengths=lengths)
            history = list(self.model.monitor_.history)
            gain = history[-1] - history[-2] if len(history) > 1 else np.inf
            self.diagnostics = {"iterations": self.model.monitor_.iter,
                                "last_log_likelihood_gain": gain,
                                "fit_observations": len(clean)}
            # hmmlearn also calls hitting n_iter 'converged'; require actual tolerance.
            if not np.isfinite(gain) or not (-1e-8 <= gain < self.config.tol):
                self.reason = "not_converged"
                return self
            cov = np.diagonal(self.model.covars_, axis1=1, axis2=2)
            arrays = [self.model.means_, cov, self.model.startprob_, self.model.transmat_]
            if not all(np.isfinite(a).all() for a in arrays):
                self.reason = "nonfinite_parameters"
                return self
            if (cov.min() < self.config.covariance_floor
                    or (cov.max(axis=1) / cov.min(axis=1)).max() > self.config.max_condition):
                self.reason = "invalid_covariance"
                return self
            if (np.any(self.model.transmat_ < 0) or np.any(self.model.startprob_ < 0)
                    or not np.allclose(self.model.transmat_.sum(axis=1), 1)
                    or not np.isclose(self.model.startprob_.sum(), 1)):
                self.reason = "invalid_transition_probabilities"
                return self
            self.valid = True
            probabilities = self.predict_filtered_proba(features)
            counts = np.bincount(probabilities.dropna().to_numpy().argmax(axis=1),
                                 minlength=self.config.n_components)
            self.diagnostics["state_observations"] = counts.tolist()
            if counts.min() < self.config.min_state_observations:
                self.valid, self.reason = False, "underpopulated_state"
                return self
            self.reason = "no_favorable_states"
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            self.valid, self.reason = False, "fit_error:" + str(exc)
        return self

    def predict_filtered_proba(self, features, initial_probability=None):
        """Pure batch causal inference; pass last posterior to continue online.

        Missing observations advance the transition but emit NaN (entry blocked).
        This method never fits or updates model/scaler parameters.
        """
        self._check_frame(features)
        k = self.config.n_components
        output = np.full((len(features), k), np.nan)
        if not self.valid:
            return pd.DataFrame(output, index=features.index, columns=range(k))
        previous = None
        if initial_probability is not None:
            initial = np.asarray(initial_probability, dtype=float)
            if initial.shape != (k,) or not np.isfinite(initial).all() or (initial < 0).any() or not np.isclose(initial.sum(), 1):
                raise ValueError("Invalid initial posterior")
            with np.errstate(divide="ignore"):
                previous = np.log(initial)
        cov = np.diagonal(self.model.covars_, axis1=1, axis2=2)
        with np.errstate(divide="ignore"):
            log_a = np.log(self.model.transmat_)
            log_start = np.log(self.model.startprob_)
        for i, row in enumerate(features.to_numpy(dtype=float)):
            prior = log_start if previous is None else logsumexp(previous[:, None] + log_a, axis=0)
            if not np.isfinite(row).all():
                previous = prior - logsumexp(prior)
                continue
            x = (row - self.scaler.mean_) / self.scaler.scale_
            with np.errstate(over="ignore", invalid="ignore"):
                emission = -.5 * (len(row) * np.log(2 * np.pi) + np.log(cov).sum(axis=1)
                                  + ((x - self.model.means_) ** 2 / cov).sum(axis=1))
            posterior = prior + emission
            norm = logsumexp(posterior)
            if not np.isfinite(norm):
                previous = prior - logsumexp(prior)
                continue
            previous = posterior - norm
            p = np.exp(previous)
            if np.isfinite(p).all() and np.isclose(p.sum(), 1, atol=1e-10):
                output[i] = p
        return pd.DataFrame(output, index=features.index, columns=range(k))

    def stream(self):
        return OnlineFilter(self)

    def decision(self, probabilities):
        p = np.asarray(probabilities, dtype=float)
        diagnostic = {"allow_long_entry": False, "current_regime": None,
                      "regime_confidence": np.nan, "p_favorable": np.nan,
                      "reason": self.reason}
        if p.shape == (self.config.n_components,) and np.isfinite(p).all() and (p >= 0).all() and np.isclose(p.sum(), 1):
            diagnostic.update(current_regime=int(p.argmax()), regime_confidence=float(p.max()),
                              p_favorable=float(p[list(sorted(self.favorable_states))].sum()))
        else:
            diagnostic["reason"] = "missing_or_invalid_features"
            return diagnostic
        if not self.valid:
            return diagnostic
        if not self.favorable_states:
            diagnostic["reason"] = "no_favorable_states"
            return diagnostic
        allowed = diagnostic["p_favorable"] >= self.config.probability_threshold
        diagnostic.update(allow_long_entry=allowed, reason="accepted" if allowed else "below_threshold")
        return diagnostic

    def allow_long_entry(self, probabilities):
        return self.decision(probabilities)["allow_long_entry"]

    def save(self, path):
        """Only load artifacts you created/trust: pickle is executable."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as file:
            pickle.dump(self, file, protocol=5)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as file:
            result = pickle.load(file)
        if not isinstance(result, cls) or result.feature_order != FEATURES:
            raise ValueError("Incompatible model artifact")
        return result

    def metadata(self):
        return {"config": asdict(self.config), "features": self.feature_order,
                "valid": self.valid, "reason": self.reason, "fit_start": self.fit_start,
                "fit_end": self.fit_end, "calibration_end": self.calibration_end,
                "favorable_states": sorted(self.favorable_states), "diagnostics": self.diagnostics}


class OnlineFilter:
    """One observation per step, preserving missing-observation transitions."""
    def __init__(self, regime_filter):
        self.filter = regime_filter
        self.posterior = None
        self.last_time = None

    def update(self, row):
        if len(row) != 1:
            raise ValueError("Online update requires one observation")
        if self.last_time is not None and row.index[0] <= self.last_time:
            raise ValueError("Online observations must advance time")
        result = self.filter.predict_filtered_proba(row, self.posterior)
        p = result.iloc[0].to_numpy()
        if np.isfinite(p).all():
            self.posterior = p
        elif self.filter.valid:
            self.posterior = (self.filter.model.startprob_.copy() if self.posterior is None
                              else self.posterior @ self.filter.model.transmat_)
        self.last_time = row.index[0]
        return result
