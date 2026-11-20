"""Optimize on the previous 26 weeks and apply each winner to the immediately following week."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
from functools import partial
import hashlib
from io import StringIO
import json
from pathlib import Path

import numpy as np
import pandas as pd

from direction_methods import DIR_METHODS, trend_at_entries
from find_best import DEFAULT_SEED
from find_best_dir import opti_dir
from read_data import ohlc_resolution, prepared_ohlc
from .config import RunConfig
from .economics import equity_curve, profit_factor
from .independent_portfolios import cross_stream, replay_week
from .market import available_index, optimize, optimizer_settings, validate_market
from .walk_forward import ROOT, code_fingerprint, write_json


def train_week(left, *, path, config, trials):
    start = left - pd.Timedelta(weeks=config.optimization_weeks)
    train = pd.read_parquet(path)
    step = ohlc_resolution(train.index)
    train = train.loc[(train.index >= start) & (train.index + step <= left)]
    message = StringIO()
    with redirect_stdout(message):
        print(f"Evaluación {left.date()} | Entrenamiento [{start}, {left}) UTC")
        print("find_best")
        ma = optimize(train, config, verbose=True)
        print("find_best_dir")
        with optimizer_settings(config):
            params, stats = opti_dir(train, n_trials=trials, seed=DEFAULT_SEED, filter_side="revert")
    return dict(entrenamiento_inicio=start, entrenamiento_fin=left, prueba_inicio=left,
                ultima_vela_entrenamiento=train.index[-1], filas_entrenamiento=len(train),
                media=ma, parametros_direccion=params, kpis_direccion=stats), message.getvalue()


def evaluate_week(data, step, trained, n, right, positions, trades):
    start, left = trained["entrenamiento_inicio"], trained["prueba_inicio"]
    params = trained["parametros_direccion"]
    control_ma = (params["ma_method"], params["ma_candle"], params["ma_lookback"])
    control = cross_stream(data, step, start, left, right, control_ma, True)
    history = data.loc[(data.index >= start) & (data.index + step <= right)]
    _, _, bars = prepared_ohlc(history, params["candle"], "h")
    trend = DIR_METHODS[params["name"]](control, params, data=bars)
    trend.index = available_index(trend.index, params["candle"], "h")
    filtered = control.loc[control.Signals.eq(-1) | ~trend_at_entries(trend, control.index)]
    original_ma = trained["media"]
    original = cross_stream(data, step, start, left, right, original_ma, True)
    for arm, current, ma in (("find_best", original, original_ma), ("find_best_dir", filtered, control_ma),
                             ("find_best_dir_sin_filtro", control, control_ma)):
        pos = positions[arm]
        carried = current.iloc[:0]
        if pos is not None:
            carried = cross_stream(data, step, pos["history_start"], left, right, pos["parameters"], True)
        positions[arm] = replay_week(pos, trades[arm], current, carried, n, start, ma)


def save_performance(out, price, capital, positions, trades):
    summary, annual = [], []
    for arm, records in trades.items():
        closed = pd.DataFrame(records, columns=["entry_time", "entry_price", "exit_time", "exit_price",
                                                "pnl", "window", "parameters", "history_start"])
        pos = positions[arm]
        opened = pd.DataFrame([] if pos is None else [pos], columns=["entry_time", "entry_price", "window",
                                                                     "parameters", "history_start"])
        curve = equity_curve(price, closed, initial_capital=capital, open_positions=opened)
        opened["mark_price"] = price.ffill().iloc[-1]
        opened["pnl"] = opened.mark_price - opened.entry_price
        eq = curve.equity
        assert np.isclose(eq.iloc[-1]-capital, closed.pnl.sum()+opened.pnl.sum())
        daily = eq.resample("1D", closed="right", label="right").last().pct_change().dropna()
        turnover = closed.entry_price.sum()+closed.exit_price.sum()+opened.entry_price.sum()
        summary.append(dict(cartera=arm, inicial=capital, final=eq.iloc[-1], retorno=eq.iloc[-1]/capital-1,
                            drawdown=(eq/eq.cummax()-1).min(), sharpe=np.sqrt(365.25)*daily.mean()/daily.std() if daily.std()>0 else np.nan,
                            exposicion=curve.units.iloc[:-1].mean(), profit_factor=profit_factor(closed.pnl),
                            aciertos=(closed.pnl>0).mean(), cerradas=len(closed), abiertas=len(opened),
                            pnl_cerrado=closed.pnl.sum(), pnl_abierto=opened.pnl.sum(), volumen_ejecutado=turnover))
        for year in range(eq.index[0].year, (eq.index[-1]-pd.Timedelta(nanoseconds=1)).year+1):
            segment = eq.loc[pd.Timestamp(year,1,1):pd.Timestamp(year+1,1,1)]
            annual.append(dict(cartera=arm, anio=year, inicial=segment.iloc[0], final=segment.iloc[-1],
                               retorno=segment.iloc[-1]/segment.iloc[0]-1))
        closed.to_csv(out/f"{arm}_trades.csv", index=False)
        opened.to_csv(out/f"{arm}_open_positions.csv", index=False)
        curve.to_csv(out/f"{arm}_equity.csv", index_label="available_at")
    pd.DataFrame(summary).to_csv(out/"comparacion.csv", index=False)
    pd.DataFrame(annual).to_csv(out/"anual.csv", index=False)


def run(data_path, reference, out, workers, trials=100):
    previous = json.loads(reference.read_text())
    config = RunConfig(**previous["run_config"])
    start, end = map(pd.Timestamp, (previous["evaluation_start"], previous["evaluation_end"]))
    source = json.loads(reference.with_name("data_source.json").read_text())
    if hashlib.sha256(data_path.read_bytes()).hexdigest() != source["sha256"]:
        raise ValueError("BTC data must match the previous comparison")
    data = pd.read_parquet(data_path)
    validate_market(data)
    step = ohlc_resolution(data.index)
    if step != pd.Timedelta(hours=1) or (config.units,config.fee_bps,config.temporality)!=(1,0,"h"):
        raise ValueError("This experiment requires hourly prices, one unit and zero fees")
    if data.index[0]>start-pd.Timedelta(weeks=config.optimization_weeks) or data.index[-1]+step<end:
        raise ValueError("Insufficient history for the requested schedule")
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/"data_source.json",source)
    weeks = pd.date_range(start, end, freq="7D", inclusive="left")
    assert list(weeks) == [pd.Timestamp(w["calibration_end"]) for w in previous["windows"]]
    positions = dict.fromkeys(("find_best","find_best_dir","find_best_dir_sin_filtro"))
    trades = {arm: [] for arm in positions}
    price = data.close.set_axis(data.index+step).reindex(pd.date_range(start,end,freq=step))
    records = []
    for_week = partial(train_week, path=data_path, config=config, trials=trials)
    with ProcessPoolExecutor(max_workers=workers) as pool, (out/"optimizadores.log").open("w") as log:
        for n, (trained, message) in enumerate(pool.map(for_week, weeks)):
            left = trained["prueba_inicio"]
            right = min(left + pd.Timedelta(weeks=1), end)
            trained.update(ventana=n, prueba_fin=right)
            records.append(trained)
            print(f"Ventana {n:03d} | {message}", file=log, flush=True)
            evaluate_week(data, step, trained, n, right, positions, trades)
            print(f"{n+1}/{len(weeks)} {left.date()} -> {right.date()}", flush=True)
    save_performance(out, price, config.initial_capital, positions, trades)
    write_json(out/"manifest.json", dict(evaluacion_inicio=start,evaluacion_fin=end,
               semanas_entrenamiento=config.optimization_weeks,desfase_semanas=0,
               semilla=DEFAULT_SEED,find_best_llamadas=config.optimizer_calls,find_best_dir_ensayos=trials,
               fuente_sha256=source["sha256"],codigo_sha256=code_fingerprint(),ventanas=records))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT/"data/BTC_hourly_2024_2025.parquet")
    parser.add_argument("--reference", type=Path, default=ROOT/"results/btc_ab_2024_2025/manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT/"results/inmediato_26s_2024_2025")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    run(args.data, args.reference, args.output, args.workers)
