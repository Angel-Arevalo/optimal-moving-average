"""Four entry policies on the same find_best_dir moving average and reference trades."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform

import numpy as np
import pandas as pd

from direction_methods import DIR_METHODS, trend_at_entries
from find_best import DEFAULT_SEED
from find_best_dir import opti_dir
from read_data import ohlc_resolution, prepared_ohlc
from .config import FeatureConfig, MarkovConfig, RunConfig
from .economics import calibrate, comparison_metrics, equity_curve, trades_from_signals
from .execution import ContinuousExecution, completed_crossing_pairs
from .market import available_index, observations, optimizer_settings, strategy_view, validate_market
from .model import MarkovRegimeFilter
from .walk_forward import ROOT, code_fingerprint, versions, write_json

BOUNDARIES = ("optimization_start", "optimization_end", "fit_end", "calibration_end", "evaluation_end")


def executable_strategy_view(history, ma_params, temporality, step):
    ma, signals = strategy_view(history, ma_params, temporality)
    return ma, signals.loc[signals.index.isin(history.index + step)]


def prepare_window(window, *, data_path, config, markov, trials, seed, output):
    number = window["window"]
    opt_start, opt_end, fit_end, test_start, test_end = [pd.Timestamp(window[k]) for k in BOUNDARIES]
    folder = output / f"window_{number:03d}"
    folder.mkdir(exist_ok=True)
    data = pd.read_parquet(data_path)
    step = ohlc_resolution(data.index)
    history = data.loc[(data.index >= opt_start) & (data.index + step <= test_end)]
    training = history.loc[history.index + step <= opt_end]
    with (folder / "optimizador.log").open("w") as log, redirect_stdout(log), optimizer_settings(config):
        print(f"Ventana {number:03d} | Optimización {opt_start.date()} a {opt_end.date()} "
              f"| Evaluación {test_start.date()} a {test_end.date()} (fines exclusivos)")
        params, stats = opti_dir(training, n_trials=trials, seed=seed, filter_side="revert")
    return fit_selected_window(window, history, step, params, stats, config, markov, output)


def fit_selected_window(window, history, step, params, stats, config, markov, output):
    number = window["window"]
    folder = output / f"window_{number:03d}"
    opt_end, fit_end, test_start, test_end = [pd.Timestamp(window[k]) for k in BOUNDARIES[1:]]
    ma_params = (params["ma_method"], params["ma_candle"], params["ma_lookback"])
    ma, signals = executable_strategy_view(history, ma_params, config.temporality, step)
    _, features = observations(history, ma, markov.features)
    features = features.loc[features.index > opt_end]
    model = MarkovRegimeFilter(markov).fit(features.loc[:fit_end], available_by=fit_end)
    probabilities = model.predict_filtered_proba(features)

    pairs = completed_crossing_pairs(signals.loc[:test_start])
    trades = trades_from_signals(pairs, units=config.units, fee_bps=config.fee_bps)
    cal_p = probabilities.loc[(probabilities.index > fit_end) & (probabilities.index <= test_start)]
    selection, _ = calibrate(model, trades, cal_p, start=fit_end, end=test_start)
    selection.to_csv(folder / "regime_selection.csv", index=False)
    model.save(output / "models" / f"window_{number:03d}.pkl")
    evaluation = probabilities.loc[probabilities.index > test_start].copy()
    _, _, detector_bars = prepared_ohlc(history, params["candle"], config.temporality)
    condition = DIR_METHODS[params["name"]](signals, params, data=detector_bars)
    condition.index = available_index(condition.index, params["candle"], config.temporality)
    evaluation["direction_allowed"] = ~trend_at_entries(condition, evaluation.index)
    signals = signals.loc[(signals.index > test_start) & (signals.index <= test_end)]
    evidence = evaluation.join(signals, how="outer")
    evidence.to_csv(folder / "evaluation.csv", index_label="available_at")
    record = dict(window, ma_parameters=ma_params, optimizer_params=params,
                  optimization_kpis=stats, model=model.metadata())
    write_json(folder / "model.json", record)
    return record


def apply_gates(decisions, direction):
    result = decisions.copy()
    result["hmm_allowed"] = decisions.allow_long_entry
    result["direction_allowed"] = direction.loc[pd.DatetimeIndex(decisions.entry_time)].to_numpy(dtype=bool)
    return {
        "media": np.ones(len(result), dtype=bool),
        "direccion": result.direction_allowed.to_numpy(),
        "hmm": result.hmm_allowed.to_numpy(),
        "direccion_hmm": (result.direction_allowed & result.hmm_allowed).to_numpy(),
    }, result


def evaluate(data, records, output, config, markov):
    step = ohlc_resolution(data.index)
    execution = ContinuousExecution(units=config.units, fee_bps=config.fee_bps, n_components=markov.n_components)
    direction = []
    for record in records:
        number = record["window"]
        left, right = map(pd.Timestamp, (record["calibration_end"], record["evaluation_end"]))
        folder = output / f"window_{number:03d}"
        evidence = pd.read_csv(folder / "evaluation.csv", index_col="available_at", parse_dates=True,
                               float_precision="round_trip")
        current = evidence[["Signals", "Prices"]].dropna()
        direction.append(evidence.direction_allowed.dropna())
        p = evidence[[str(i) for i in range(markov.n_components)]]
        model = MarkovRegimeFilter.load(output / "models" / f"window_{number:03d}.pkl")
        carried = current.iloc[:0]
        if execution.position is not None:
            pos = execution.position
            history = data.loc[(data.index >= pos["history_start"]) & (data.index + step <= right)]
            _, carried = executable_strategy_view(history, pos["parameters"], config.temporality, step)
        execution.advance(current, carried, p, model, start=left, end=right, window=number,
                          parameters=record["ma_parameters"], history_start=pd.Timestamp(record["optimization_start"]))

    start, end = pd.Timestamp(records[0]["calibration_end"]), pd.Timestamp(records[-1]["evaluation_end"])
    price = data.close.set_axis(data.index + step).reindex(pd.date_range(start, end, freq=step))
    gates, candidates = apply_gates(execution.decision_frame("HMM"), pd.concat(direction))
    candidates.rename(columns={"reason":"hmm_reason"}).drop(columns="allow_long_entry").to_csv(
        output / "decisiones.csv", index=False)
    base_trades, base_open = execution.trades("Original"), execution.open_positions("Original")
    metrics, annual = [], []
    for name, mask in gates.items():
        decisions = candidates.assign(allow_long_entry=mask)
        times = decisions.loc[mask, "entry_time"]
        trades = base_trades.loc[base_trades.entry_time.isin(times)].drop(columns="accepted")
        opened = base_open.loc[base_open.entry_time.isin(times)].drop(columns="accepted").copy()
        assert len(trades) + len(opened) == len(times)
        curve = equity_curve(price, trades, initial_capital=config.initial_capital, units=config.units,
                             fee_bps=config.fee_bps, open_positions=opened)
        stats = comparison_metrics(name, curve, trades, decisions, config.initial_capital)
        opened["mark_price"] = price.ffill().iloc[-1]
        opened["marked_at"] = end
        opened["unrealized_pnl_after_entry_fee"] = config.units * (
            opened.mark_price - opened.entry_price * (1 + config.fee_bps / 10000))
        assert np.isclose(stats["Net PnL"], trades.pnl.sum() + opened.unrealized_pnl_after_entry_fee.sum())
        metrics.append(stats)
        trades.to_csv(output / f"{name}_trades.csv", index=False)
        opened.to_csv(output / f"{name}_open_positions.csv", index=False)
        curve.to_csv(output / f"{name}_equity.csv", index_label="available_at")
        for year in range(start.year, (end - pd.Timedelta(nanoseconds=1)).year + 1):
            left, right = max(start, pd.Timestamp(year, 1, 1)), min(end, pd.Timestamp(year+1, 1, 1))
            eq = curve.loc[left:right, "equity"]
            annual.append(dict(anio=year, estrategia=name, capital_inicial=eq.iloc[0], capital_final=eq.iloc[-1],
                               retorno=eq.iloc[-1]/eq.iloc[0]-1, drawdown=(eq/eq.cummax()-1).min(),
                               operaciones_cerradas=int(((trades.exit_time > left) & (trades.exit_time <= right)).sum())))
    result = pd.DataFrame(metrics)
    result.to_csv(output / "comparison.csv", index=False)
    pd.DataFrame(annual).to_csv(output / "resultados_anuales.csv", index=False)
    return result


def run(data_path, reference, output, *, trials=100, seed=DEFAULT_SEED, workers=1):
    manifest = json.loads(reference.read_text())
    config = RunConfig(**manifest["run_config"])
    hmm_config = manifest["markov_config"].copy()
    hmm_config["features"] = FeatureConfig(**hmm_config["features"])
    markov = replace(MarkovConfig(**hmm_config), enabled=True)
    windows = [{key: w[key] for key in ("window", *BOUNDARIES)} for w in manifest["windows"]]
    source = json.loads(reference.with_name("data_source.json").read_text())
    if hashlib.sha256(data_path.read_bytes()).hexdigest() != source["sha256"]:
        raise ValueError("The source data must match the reference run")
    data = pd.read_parquet(data_path)
    validate_market(data)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "data_source.json", source)
    prepare = partial(prepare_window, data_path=data_path, config=config, markov=markov,
                      trials=trials, seed=seed, output=output)
    records = []
    with ProcessPoolExecutor(max_workers=workers) as pool, (output / "optimizadores.log").open("w") as log:
        for record in pool.map(prepare, windows):
            number = record["window"]
            report = (output / f"window_{number:03d}" / "optimizador.log").read_text()
            log.write(report)
            log.flush()
            print(report, flush=True)
            records.append(record)
    pd.DataFrame([{"window": w["window"], **{k:w[k] for k in BOUNDARIES},
                   **w["optimizer_params"], **w["optimization_kpis"]} for w in records]).to_csv(
        output / "kpis_optimizacion_semanal.csv", index=False)
    result = evaluate(data, records, output, config, markov)
    write_json(output / "manifest.json", dict(
        reference=reference.name, reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
        evaluation_start=records[0]["calibration_end"], evaluation_end=records[-1]["evaluation_end"],
        run_config=asdict(config), markov_config=asdict(markov), optimizer="find_best_dir.opti_dir",
        optimizer_trials=trials, seed=seed, filter_side="revert", windows=records,
        versions=dict(versions(), optuna=importlib.metadata.version("optuna")),
        python=platform.python_version(), code_sha256=code_fingerprint(),
        note="Weekly optimization on the reference date partitions. Discard trend-classified longs. "
             "Shared HMM calibrated on unfiltered natural trades of the selected moving average. "
             "Calibration and evaluation require an observed source close for fills."))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT/"data/BTC_hourly_2024_2025.parquet")
    parser.add_argument("--reference", type=Path, default=ROOT/"results/btc_ab_2024_2025/manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT/"results/btc_direccion_hmm_2024_2025")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    print(run(args.data, args.reference, args.output, trials=args.trials, seed=args.seed,
              workers=args.workers).to_string(index=False))
