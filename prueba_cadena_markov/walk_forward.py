"""Disjoint MA selection / HMM fit / economic calibration / weekly evaluation."""
from dataclasses import asdict, replace
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import numpy as np
import pandas as pd

from read_data import ohlc_resolution
from .config import MarkovConfig, RunConfig
from .market import optimize, strategy_view, observations, validate_market
from .model import MarkovRegimeFilter
from .economics import (trades_from_signals, contained_trades, calibrate,
                        state_summary, equity_curve, comparison_metrics, annotate_trades)
from .execution import ContinuousExecution, completed_crossing_pairs

ROOT = Path(__file__).resolve().parent


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str))


def code_fingerprint():
    files = sorted(ROOT.glob("*.py")) + sorted(ROOT.parent.glob("*.py"))
    return hashlib.sha256(b"".join(f.name.encode() + f.read_bytes() for f in files)).hexdigest()


def fingerprint(frame, settings):
    payload = pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()
    payload += json.dumps(settings, sort_keys=True, default=str).encode()
    payload += str(tuple(frame.columns)).encode()
    payload += code_fingerprint().encode()
    payload += json.dumps(versions(), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def versions():
    return {name: importlib.metadata.version(name) for name in
            ("numpy", "pandas", "scipy", "scikit-learn", "hmmlearn", "scikit-optimize", "ta-lib", "qnt")}


def cached_fit(features, config, cutoff, cache):
    key = fingerprint(features, {"config": asdict(config), "cutoff": cutoff})
    path = Path(cache) / ("hmm_" + key + ".pkl")
    if path.exists():
        return MarkovRegimeFilter.load(path)
    model = MarkovRegimeFilter(config).fit(features, available_by=cutoff)
    model.save(path)
    return model


def calibration_experiments(features, all_trades, start, end, fit_end, config, cache):
    """Explore C/D ONLY in historical calibration, never using test performance.

    First half selects favorable states; second half reports candidate sensitivity.
    No candidate replaces the fixed A/B policy automatically.
    """
    midpoint = start + (end-start)/2
    rows = []
    for count in (2, 3, 4):
        candidate_config = replace(config, n_components=count)
        model = cached_fit(features.loc[features.index <= fit_end], candidate_config, fit_end, cache)
        p = model.predict_filtered_proba(features.loc[features.index <= end])
        select_p = p.loc[(p.index > start) & (p.index <= midpoint)]
        calibrate(model, all_trades, select_p, start=start, end=midpoint)
        validation = contained_trades(all_trades, midpoint, end)
        validation = annotate_trades(validation, p)
        for threshold in (.55, .65, .70, .75, .85):
            prob_columns = [f"p_{s}" for s in sorted(model.favorable_states)]
            pf = validation[prob_columns].sum(axis=1) if prob_columns else pd.Series(0., index=validation.index)
            chosen = validation.loc[(pf >= threshold) & model.valid]
            rows.append(dict(n_components=count, threshold=threshold, phase="calibration_validation",
                             selection_end=midpoint, validation_end=end, favorable_states=sorted(model.favorable_states),
                             model_valid=model.valid, reason=model.reason, number_of_trades=len(chosen),
                             net_pnl=chosen.pnl.sum(), average_net_return=chosen.net_return.mean(),
                             original_signals=len(validation), accepted_signals=len(chosen),
                             sufficient_evidence=bool(len(chosen) >= config.min_trades_per_regime)))
    return pd.DataFrame(rows)


def run_walk_forward(data, evaluation_start, run_config=RunConfig(), markov_config=MarkovConfig(),
                     *, output=None, experiments=False):
    validate_market(data)
    output = Path(output) if output else ROOT / "results" / "btc_ab"
    output.mkdir(parents=True, exist_ok=True)
    cache = ROOT / "cache" / "models"
    cache.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp(evaluation_start)
    if start.dayofweek != 0 or start != start.normalize():
        raise ValueError("Evaluation starts Monday 00:00 UTC for the retraining schedule")
    source = ohlc_resolution(data.index)
    end = start + pd.Timedelta(weeks=run_config.evaluation_weeks)
    earliest = start - pd.Timedelta(weeks=run_config.optimization_weeks + run_config.fit_weeks + run_config.calibration_weeks)
    if data.index[0] > earliest or data.index[-1] + source < end:
        raise ValueError(f"Insufficient data: need source bars covering {earliest} through {end}")
    summaries, selection_rows, windows, sensitivity = [], [], [], []
    evaluation_prices = []
    evaluation_contexts = []
    execution = ContinuousExecution(units=run_config.units, fee_bps=run_config.fee_bps,
                                    n_components=markov_config.n_components)
    frozen_config = replace(markov_config, enabled=True)
    for number in range(run_config.evaluation_weeks):
        test_start = start + pd.Timedelta(weeks=number)
        test_end = test_start + pd.Timedelta(weeks=1)
        fit_end = test_start - pd.Timedelta(weeks=run_config.calibration_weeks)
        opt_end = fit_end - pd.Timedelta(weeks=run_config.fit_weeks)
        opt_start = opt_end - pd.Timedelta(weeks=run_config.optimization_weeks)
        optimization_data = data.loc[(data.index >= opt_start) & (data.index + source <= opt_end)]
        key = fingerprint(optimization_data, asdict(run_config))
        param_path = cache / ("ma_" + key + ".json")
        if param_path.exists():
            parameters = tuple(json.loads(param_path.read_text()))
        else:
            parameters = optimize(optimization_data, run_config)
            write_json(param_path, parameters)
        # Whole completed weeks, with past warmup. Only historical configuration is used.
        available_data = data.loc[(data.index >= opt_start) & (data.index + source <= test_end)]
        ma, signals = strategy_view(available_data, parameters, run_config.temporality)
        price, features = observations(available_data, ma, frozen_config.features)
        fit_features = features.loc[(features.index > opt_end) & (features.index <= fit_end)]
        model = cached_fit(fit_features, frozen_config, fit_end, cache)
        causal_features = features.loc[(features.index > opt_end) & (features.index <= test_end)]
        probabilities = model.predict_filtered_proba(causal_features)
        original_trades = trades_from_signals(completed_crossing_pairs(signals),
                                              units=run_config.units, fee_bps=run_config.fee_bps)
        calibration_p = probabilities.loc[(probabilities.index > fit_end) & (probabilities.index <= test_start)]
        selection, calibration_trades = calibrate(model, original_trades, calibration_p,
                                                 start=fit_end, end=test_start)
        selection["window"] = number
        selection["phase"] = "calibration"
        selection_rows.append(selection)
        if experiments:
            experiment = calibration_experiments(causal_features.loc[:test_start], original_trades,
                                                  fit_end, test_start, fit_end, frozen_config, cache)
            experiment["window"] = number
            sensitivity.append(experiment)
        window_dir = output / f"window_{number:03d}"
        window_dir.mkdir(exist_ok=True)
        model.save(output / "models" / f"window_{number:03d}.pkl")
        calibration_trades.to_csv(window_dir / "calibration_trades.csv", index=False)
        window = dict(window=number, optimization_start=opt_start, optimization_end=opt_end,
                      fit_end=fit_end, calibration_end=test_start, evaluation_end=test_end,
                      ma_parameters=parameters, model=model.metadata(),
                      observation_frequency=str(source), optimization_hash=key,
                      fit_hash=fingerprint(fit_features, asdict(frozen_config)))
        windows.append(window)
        write_json(window_dir / "model.json", window)
        for phase, left, right in (("fit_descriptive", opt_end, fit_end), ("calibration", fit_end, test_start)):
            phase_features = causal_features.loc[(causal_features.index > left) & (causal_features.index <= right)]
            phase_p = probabilities.reindex(phase_features.index)
            phase_trades = contained_trades(original_trades, left, right)
            summary = state_summary(phase_features, phase_p, phase_trades,
                                    price.reindex(phase_features.index), model)
            summary["window"], summary["phase"], summary["strategy"] = number, phase, "Original"
            summaries.append(summary)
            annotate_trades(phase_trades, phase_p).to_csv(window_dir / f"{phase}_trades.csv", index=False)
        phase_p = probabilities.loc[(probabilities.index > test_start) & (probabilities.index <= test_end)]
        phase_p.to_csv(window_dir / "evaluation_probabilities.csv", index_label="available_at")
        evaluation_prices.append(price.loc[(price.index >= test_start) & (price.index <= test_end)])
        carried_signals = signals.iloc[:0]
        if execution.position is not None:
            held = execution.position
            held_data = data.loc[(data.index >= held["history_start"]) & (data.index+source <= test_end)]
            _, carried_signals = strategy_view(held_data, held["parameters"], run_config.temporality)
        execution.advance(signals, carried_signals, phase_p, model, start=test_start, end=test_end,
                          window=number, parameters=parameters, history_start=opt_start)
        evaluation_contexts.append((number, causal_features.reindex(phase_p.index), phase_p,
                                    price.reindex(phase_p.index), model))
        print(f"Window {number}: {test_start.date()}, MA={parameters}, valid={model.valid}, "
              f"states={sorted(model.favorable_states)}, reason={model.reason}", flush=True)
    price = pd.concat(evaluation_prices)
    price = price.loc[~price.index.duplicated()].sort_index()
    metrics = []
    for name in ("Original", "HMM"):
        trades = execution.trades(name)
        decisions = execution.decision_frame(name)
        open_positions = execution.open_positions(name)
        curve = equity_curve(price, trades, initial_capital=run_config.initial_capital,
                             units=run_config.units, fee_bps=run_config.fee_bps,
                             open_positions=open_positions)
        metrics.append(comparison_metrics(name, curve, trades, decisions, run_config.initial_capital))
        slug = name.lower()
        trades.to_csv(output / f"{slug}_trades.csv", index=False)
        decisions.to_csv(output / f"{slug}_decisions.csv", index=False)
        curve.to_csv(output / f"{slug}_equity.csv", index_label="available_at")
        open_positions["marked_at"] = price.index[-1]
        open_positions["mark_price"] = price.ffill().iloc[-1]
        open_positions["unrealized_pnl_after_entry_fee"] = run_config.units * (
            open_positions.mark_price - open_positions.entry_price * (1+run_config.fee_bps/10000))
        open_positions.to_csv(output / f"{slug}_open_positions.csv", index=False)
        for number, eval_features, eval_p, eval_price, eval_model in evaluation_contexts:
            window_trades = trades.loc[trades.window == number]
            window_decisions = decisions.loc[decisions.window == number]
            summary = state_summary(eval_features, eval_p, window_trades, eval_price, eval_model)
            summary["number_of_signals"] = [int((window_decisions.current_regime == s).sum()) for s in summary.regime]
            summary["window"], summary["phase"], summary["strategy"] = number, "evaluation", name
            summaries.append(summary)
            filename = "evaluation_trades.csv" if name == "Original" else "evaluation_hmm_trades.csv"
            window_trades.to_csv(output / f"window_{number:03d}" / filename, index=False)
    result = pd.DataFrame(metrics)
    result.to_csv(output / "comparison.csv", index=False)
    pd.concat(summaries, ignore_index=True).to_csv(output / "regimes.csv", index=False)
    pd.concat(selection_rows, ignore_index=True).to_csv(output / "regime_selection.csv", index=False)
    if sensitivity:
        pd.concat(sensitivity, ignore_index=True).to_csv(output / "calibration_sensitivity.csv", index=False)
    write_json(output / "manifest.json", dict(run_config=asdict(run_config), markov_config=asdict(frozen_config),
               evaluation_start=start, evaluation_end=end, windows=windows, versions=versions(),
               python=platform.python_version(), code_sha256=code_fingerprint(),
               market_sha256=fingerprint(data, {}), source_rows=len(data),
               note="No weekly or terminal liquidation. Entry MA and its history origin frozen until natural exit. "
                    "Open positions marked to market; trade stats use completed trades. "
                    "Optimizer retains legacy objective/closures. A/B share reference entry candidates. "
                    "Sensitivity never selects on test. Regime IDs and trade attribution use entry window."))
    return result
