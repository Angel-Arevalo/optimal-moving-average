"""Retrospective comparison of two frozen find_best selection rules using trailing net P&L."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from .economics import equity_curve
from .independent_portfolios import cross_stream, replay_week
from .walk_forward import ROOT

FEE_BPS = 5.
CAPITAL = 100000.
VALIDATION = pd.Timedelta(weeks=13)
START, END = pd.Timestamp("2025-01-06"), pd.Timestamp("2026-01-01")


def net_curve(price, trades, opened):
    return equity_curve(price, trades, fee_bps=FEE_BPS, initial_capital=CAPITAL, open_positions=opened).equity


def shadow_curve(price, folder, slug):
    trades = pd.read_csv(folder / f"{slug}_trades.csv", parse_dates=["entry_time", "exit_time"])
    opened = pd.read_csv(folder / f"{slug}_open_positions.csv", parse_dates=["entry_time"])
    return net_curve(price, trades, opened)


def main():
    results = ROOT / "results"
    out = results / "validacion_13s_find_best_2025"
    data = pd.read_parquet(ROOT / "data/BTC_hourly_2024_2025.parquet")
    step = pd.Timedelta(hours=1)
    price = data.close.set_axis(data.index + step).reindex(pd.date_range("2024-01-01", END, freq=step))
    immediate = results / "inmediato_26s_2024_2025"
    lagged = results / "btc_ab_2024_2025"
    recent = json.loads((immediate / "manifest.json").read_text())["ventanas"]
    old = json.loads((lagged / "manifest.json").read_text())["windows"]
    shadow = {"inmediata": shadow_curve(price, immediate, "find_best"),
              "desfase_78": shadow_curve(price, lagged, "original")}
    strategies = ["inmediata", "desfase_78", "selector"]
    positions = dict.fromkeys(strategies)
    trades = {key: [] for key in strategies}
    decisions = []
    winner = "inmediata"
    for n, (r, o) in enumerate(zip(recent, old, strict=True)):
        left, right = map(pd.Timestamp, (r["prueba_inicio"], r["prueba_fin"]))
        assert (left, right) == (pd.Timestamp(o["calibration_end"]), pd.Timestamp(o["evaluation_end"]))
        if left < START:
            continue
        pnl = {key: curve.at[left] - curve.at[left - VALIDATION] for key, curve in shadow.items()}
        if pnl["inmediata"] != pnl["desfase_78"]:
            winner = max(pnl, key=pnl.get)
        decisions.append(dict(inicio_validacion=left-VALIDATION, fin_validacion=left, inicio_prueba=left,
                              fin_prueba=right, elegida=winner, **pnl))
        configs = {"inmediata": (pd.Timestamp(r["entrenamiento_inicio"]), tuple(r["media"])),
                   "desfase_78": (pd.Timestamp(o["optimization_start"]), tuple(o["ma_parameters"]))}
        for key in strategies:
            warm, ma = configs[winner if key == "selector" else key]
            signals = cross_stream(data, step, warm, left, right, ma, True)
            pos = positions[key]
            carried = signals.iloc[:0] if pos is None else cross_stream(
                data, step, pos["history_start"], left, right, pos["parameters"], True)
            positions[key] = replay_week(pos, trades[key], signals, carried, n, warm, ma)
    out.mkdir(exist_ok=False)
    pd.DataFrame(decisions).to_csv(out / "decisiones.csv", index=False)
    summary = []
    for key in strategies:
        closed = pd.DataFrame(trades[key])
        opened = pd.DataFrame([] if positions[key] is None else [positions[key]], columns=["entry_time", "entry_price"])
        eq = net_curve(price.loc[START:END], closed, opened)
        volume = closed.entry_price.sum() + closed.exit_price.sum() + opened.entry_price.sum()
        gross = closed.pnl.sum() + (price.at[END] - opened.entry_price).sum()
        assert np.isclose(eq.iloc[-1], CAPITAL + gross - volume*FEE_BPS/10000)
        summary.append(dict(estrategia=key, inicio=START, fin_exclusivo=END, semanas_validacion=13,
                            coste_pb_por_lado=FEE_BPS, operaciones_cerradas=len(closed),
                            retorno_neto=eq.iloc[-1]/CAPITAL-1, drawdown=(eq/eq.cummax()-1).min()))
    print(pd.DataFrame(summary).to_string(index=False))
    pd.DataFrame(summary).to_csv(out / "comparacion.csv", index=False)


if __name__ == "__main__":
    main()
