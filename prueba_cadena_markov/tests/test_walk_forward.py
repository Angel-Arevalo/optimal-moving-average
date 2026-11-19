from dataclasses import replace
import json
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from prueba_cadena_markov.config import RunConfig, MarkovConfig
from prueba_cadena_markov.model import MarkovRegimeFilter
from prueba_cadena_markov import walk_forward as wf


def data_for_integration():
    idx = pd.date_range("2023-01-02", periods=24*7*10, freq="h")
    rng = np.random.default_rng(7)
    p = 100+np.sin(np.arange(len(idx))/5)*3 + rng.normal(0, .3, len(idx))
    return pd.DataFrame({"open": p, "high": p+1, "low": p-1, "close": p}, index=idx)


def test_real_optimizer_still_runs_and_restores_globals():
    import keys
    from prueba_cadena_markov.market import optimize
    before = keys.temp, keys.calls, keys.mid_cache
    data = data_for_integration().iloc[:24*7*2]
    config = RunConfig(optimizer_calls=10, candle_max=2, lookback_max=6, methods=("SMA", "EMA"))
    params = optimize(data, config)
    assert params[0] in config.methods and params[1] in (1, 2) and 2 <= params[2] <= 6
    assert before == (keys.temp, keys.calls, keys.mid_cache)


def test_walk_forward_separation_prefix_and_accounting(tmp_path, monkeypatch):
    data = data_for_integration()
    test_start = data.index[0] + pd.Timedelta(weeks=8)
    cfg = RunConfig(optimization_weeks=2, fit_weeks=2, calibration_weeks=4, evaluation_weeks=2,
                    optimizer_calls=10, candle_max=2, lookback_max=6)
    mk = MarkovConfig(min_observations=100, min_state_observations=5, min_trades_per_regime=5)
    monkeypatch.setattr(wf, "ROOT", tmp_path)
    optimization_inputs, fit_inputs = [], []
    def optimize_spy(frame, config):
        optimization_inputs.append(frame.copy())
        return "SMA", 2, 4
    actual_fit = MarkovRegimeFilter.fit
    def fit_spy(self, features, *, available_by):
        fit_inputs.append((features.copy(), available_by))
        return actual_fit(self, features, available_by=available_by)
    monkeypatch.setattr(wf, "optimize", optimize_spy)
    monkeypatch.setattr(MarkovRegimeFilter, "fit", fit_spy)
    output = tmp_path / "first"
    result = wf.run_walk_forward(data, test_start, cfg, mk, output=output, experiments=True)
    manifest = json.loads((output / "manifest.json").read_text())
    assert len(optimization_inputs) == 2
    for window, optimized in zip(manifest["windows"], optimization_inputs):
        opt_end = pd.Timestamp(window["optimization_end"])
        fit_end = pd.Timestamp(window["fit_end"])
        calibration_end = pd.Timestamp(window["calibration_end"])
        assert optimized.index.max()+pd.Timedelta(hours=1) <= opt_end < fit_end < calibration_end
    for frame, cutoff in fit_inputs:
        assert frame.index.max() <= cutoff < test_start
        assert frame.index.min() > cutoff-pd.Timedelta(weeks=2)
    sensitivity = pd.read_csv(output / "calibration_sensitivity.csv")
    assert len(sensitivity) == 2*3*5
    assert pd.to_datetime(sensitivity.validation_end).max() <= test_start+pd.Timedelta(weeks=1)
    for name in ("original", "hmm"):
        trades = pd.read_csv(output / f"{name}_trades.csv")
        curve = pd.read_csv(output / f"{name}_equity.csv")
        opened = pd.read_csv(output / f"{name}_open_positions.csv")
        assert np.isclose(curve.equity.iloc[-1]-cfg.initial_capital,
                          trades.pnl.sum()+opened.unrealized_pnl_after_entry_fee.sum())
        assert curve.units.min() >= 0
    # Mutate all future weeks, then rerun a one-week prefix. The first window's
    # configurations, model, posterior and decisions must remain unchanged.
    changed = data.copy()
    changed.loc[changed.index >= test_start+pd.Timedelta(weeks=1)] *= 100
    prefix = tmp_path / "prefix"
    wf.run_walk_forward(changed, test_start, replace(cfg, evaluation_weeks=1), mk, output=prefix)
    for filename in ("evaluation_probabilities.csv",):
        assert_frame_equal(pd.read_csv(output / "window_000" / filename),
                           pd.read_csv(prefix / "window_000" / filename))
    decisions = pd.read_csv(output / "hmm_decisions.csv")
    first_decisions = decisions.loc[decisions.window == 0].reset_index(drop=True)
    assert_frame_equal(first_decisions, pd.read_csv(prefix / "hmm_decisions.csv"))
    assert result.loc[1, "Number of Trades"] <= result.loc[0, "Number of Trades"]
    for name in ("original", "hmm"):
        full = pd.read_csv(output / f"{name}_equity.csv", index_col=0)
        short = pd.read_csv(prefix / f"{name}_equity.csv", index_col=0)
        assert_frame_equal(full.loc[short.index], short)


def test_missing_boundary_bar_does_not_require_liquidation(tmp_path, monkeypatch):
    data = data_for_integration()
    start = data.index[0] + pd.Timedelta(weeks=8)
    data = data.drop(start-pd.Timedelta(hours=1))
    cfg = RunConfig(optimization_weeks=2, fit_weeks=2, calibration_weeks=4, evaluation_weeks=1)
    monkeypatch.setattr(wf, "ROOT", tmp_path)
    monkeypatch.setattr(wf, "optimize", lambda *args: ("SMA", 2, 4))
    wf.run_walk_forward(data, start, cfg, output=tmp_path / "result")
    trades = pd.read_csv(tmp_path / "result/original_trades.csv")
    assert set(trades.exit_reason) <= {"ma_cross"}


def test_cache_is_sensitive_to_data_parameters_and_cutoff(tmp_path):
    frame = pd.DataFrame({"x": [1., 2.]}, index=pd.date_range("2024-01-01", periods=2, freq="h"))
    key = wf.fingerprint(frame, {"threshold": .7})
    assert key == wf.fingerprint(frame.copy(), {"threshold": .7})
    assert key != wf.fingerprint(frame, {"threshold": .8})
    assert key != wf.fingerprint(frame*2, {"threshold": .7})
