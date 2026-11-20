"""Replay the archived weekly signals with separate positions and cash for each strategy."""
import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd

from .economics import equity_curve
from .market import strategy_view
from .walk_forward import ROOT
from read_data import ohlc_resolution


def replay_week(position, trades, entries, carried, window, warmup, parameters):
    # Rejected entries leave the account flat, including across a model change.
    for time in entries.index.union(carried.index).sort_values():
        if position is not None:
            stream = entries if position["window"] == window else carried
            if time in stream.index and stream.at[time, "Signals"] == -1:
                price = float(stream.at[time, "Prices"])
                trades.append(dict(position, exit_time=time, exit_price=price,
                                   pnl=price-position["entry_price"]))
                position = None
            continue
        if time in entries.index and entries.at[time, "Signals"] == 1:
            position = dict(entry_time=time, entry_price=float(entries.at[time, "Prices"]),
                            parameters=parameters, history_start=warmup, window=window)
    return position


def cross_stream(data, step, warmup, left, right, parameters, require_close):
    history = data.loc[(data.index >= warmup) & (data.index + step <= right)]
    _, signals = strategy_view(history, parameters)
    signals = signals.loc[(signals.index > left) & (signals.index <= right)]
    if require_close:
        signals = signals.loc[signals.index.isin(history.index + step)]
    return signals


def replay_family(data, folder, family, arms, out):
    manifest = json.loads((folder / "manifest.json").read_text())
    config = manifest["run_config"]
    if (config["units"], config["fee_bps"], config["temporality"]) != (1, 0, "h"):
        raise ValueError("This replay requires the archived one-unit, zero-fee hourly experiment")
    capital = config["initial_capital"]
    step = ohlc_resolution(data.index)
    has_direction = family == "find_best_dir"
    positions = dict.fromkeys(arms)
    trades = {arm: [] for arm in arms}
    for window in manifest["windows"]:
        n = window["window"]
        warmup, left, right = [pd.Timestamp(window[k]) for k in ("optimization_start", "calibration_end", "evaluation_end")]
        params = tuple(window["ma_parameters"])
        current = cross_stream(data, step, warmup, left, right, params, has_direction)
        filename = "evaluation.csv" if has_direction else "evaluation_probabilities.csv"
        evidence = pd.read_csv(folder / f"window_{n:03d}" / filename, index_col=0, parse_dates=True,
                               float_precision="round_trip").reindex(current.index)
        states = window["model"]["favorable_states"]
        hmm = pd.Series(False, index=current.index)
        if window["model"]["valid"] and states:
            posterior = evidence[[str(i) for i in states]]
            hmm = posterior.notna().all(axis=1) & (posterior.sum(axis=1) >= manifest["markov_config"]["probability_threshold"])
        for arm, (need_direction, need_hmm, _) in arms.items():
            allowed = pd.Series(True, index=current.index)
            if need_direction:
                allowed &= evidence.direction_allowed.eq(True)
            if need_hmm:
                allowed &= hmm
            stream = current.loc[allowed | current.Signals.eq(-1)]
            pos = positions[arm]
            carried = current.iloc[:0]
            if pos is not None:
                carried = cross_stream(data, step, pos["history_start"], left, right, pos["parameters"], has_direction)
            positions[arm] = replay_week(pos, trades[arm], stream, carried, n, warmup, params)

    start, end = [pd.Timestamp(manifest[k]) for k in ("evaluation_start", "evaluation_end")]
    price = data.close.set_axis(data.index + step).reindex(pd.date_range(start, end, freq=step))
    summary, years = [], []
    for arm, (_, _, archived_arm) in arms.items():
        slug = f"{family}_{arm}"
        closed = pd.DataFrame(trades[arm], columns=["entry_time", "entry_price", "exit_time", "exit_price",
                                                    "pnl", "parameters", "history_start", "window"])
        pos = positions[arm]
        opened = pd.DataFrame([] if pos is None else [pos], columns=["entry_time", "entry_price", "parameters",
                                                                     "history_start", "window"])
        curve = equity_curve(price, closed, initial_capital=capital, open_positions=opened)
        pnl = closed.pnl.sum() + (price.iloc[-1]-pos["entry_price"] if pos is not None else 0)
        assert np.isclose(curve.equity.iloc[-1], capital+pnl)
        archived = pd.read_csv(folder / f"{archived_arm}_trades.csv", parse_dates=["entry_time", "exit_time"])
        old_curve = pd.read_csv(folder / f"{archived_arm}_equity.csv", index_col=0, parse_dates=True)
        columns = ["entry_time", "entry_price", "exit_time", "exit_price", "pnl"]
        same = archived[columns].equals(closed[columns]) and old_curve.equals(curve)
        closed.to_csv(out / f"{slug}_trades.csv", index=False)
        opened.to_csv(out / f"{slug}_open_positions.csv", index=False)
        curve.to_csv(out / f"{slug}_equity.csv", index_label="available_at")
        summary.append(dict(optimizador=family, cartera=arm, inicial=capital, final=capital+pnl, retorno=pnl/capital,
                            operaciones_cerradas=len(closed), exposicion=curve.units.iloc[:-1].mean(),
                            drawdown=(curve.equity/curve.equity.cummax()-1).min(), misma_ejecucion_que_referencia=same))
        for year in range(start.year, (end-step).year+1):
            eq = curve.equity.loc[pd.Timestamp(year, 1, 1):pd.Timestamp(year+1, 1, 1)]
            years.append(dict(optimizador=family, cartera=arm, anio=year, retorno=eq.iloc[-1]/eq.iloc[0]-1))
    return summary, years


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT/"data/BTC_hourly_2024_2025.parquet")
    parser.add_argument("--output", type=Path, default=ROOT/"results/carteras_independientes_2024_2025")
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    data = pd.read_parquet(args.data)
    families = [
        ("find_best", "btc_ab_2024_2025", {"media": (False, False, "original"), "hmm": (False, True, "hmm")}),
        ("find_best_dir", "btc_direccion_hmm_2024_2025",
         {"media_control": (False, False, "media"), "direccion": (True, False, "direccion"),
          "hmm_control": (False, True, "hmm"), "direccion_hmm": (True, True, "direccion_hmm")}),
    ]
    summary, annual = [], []
    for family, folder, arms in families:
        rows, years = replay_family(data, ROOT/"results"/folder, family, arms, args.output)
        summary.extend(rows)
        annual.extend(years)
    pd.DataFrame(summary).to_csv(args.output/"comparacion.csv", index=False)
    pd.DataFrame(annual).to_csv(args.output/"anual.csv", index=False)


if __name__ == "__main__":
    main()
