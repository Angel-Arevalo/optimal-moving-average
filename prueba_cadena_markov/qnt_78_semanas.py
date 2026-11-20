"""Replay find_best's 78-week gap with Quantiacs and a USD 10 fee per order of 1 BTC."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import qnt.stats as qs

from .independent_portfolios import cross_stream, replay_week

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "results/btc_ab_2024_2025"
OUTPUT = ROOT / "results/qnt_78s_10usd_2024_2025"
HOUR = pd.Timedelta(hours=1)
FEE = 10.0


def replay_orders(market, windows):
    closed, position = [], None
    for w in windows:
        warmup, opt_end, left, right = [pd.Timestamp(w[k]) for k in
            ("optimization_start", "optimization_end", "calibration_end", "evaluation_end")]
        if left - opt_end != pd.Timedelta(weeks=78):
            raise ValueError("Unexpected gap after MA optimization")
        params = tuple(w["ma_parameters"])
        entries = cross_stream(market, HOUR, warmup, left, right, params, False)
        carried = entries.iloc[:0] if position is None else cross_stream(
            market, HOUR, position["history_start"], left, right, position["parameters"], False)
        position = replay_week(position, closed, entries, carried, w["window"], warmup, params)
    orders = []
    for trade in closed:
        orders.append((trade["entry_time"], 1, trade["entry_price"]))
        orders.append((trade["exit_time"], -1, trade["exit_price"]))
    if position is not None:
        orders.append((position["entry_time"], 1, position["entry_price"]))
    return pd.DataFrame(orders, columns=["time", "change", "price"]).set_index("time")


def simulate(mark, orders, capital):
    if not orders.index.is_unique or not orders.index.is_monotonic_increasing:
        raise ValueError("Expected at most one order per close in chronological order")
    np.testing.assert_array_equal(mark.loc[orders.index], orders.price)
    change = orders.change.reindex(mark.index, fill_value=0)
    units = change.cumsum()
    if not units.isin([0, 1]).all():
        raise ValueError("A fixed fee per order requires each change to be exactly one BTC")
    fee = change.abs() * FEE
    cash = capital - (change * mark + fee).cumsum()
    expected_equity = cash + units * mark
    # QNT derives shares from weights at execution before deducting transaction costs.
    weight = units * mark / (expected_equity + fee)
    if not weight.between(0, 1).all():
        raise ValueError("The requested position exceeds available capital")
    price = mark.to_numpy()[:, None]
    # Each point is a close-price execution event. Between events the previous position
    # earns the entire close-to-close move. USD per BTC changed equals USD per order here.
    rr = qs.calc_relative_return_np(
        weight.to_numpy()[:, None], np.ones(price.shape, dtype=bool), price, price,
        np.full_like(price, FEE), np.zeros_like(price), None, None)
    returns = xr.DataArray(rr, dims="time", coords={"time": mark.index.values})
    net = qs.calc_equity(returns).to_series() * capital
    np.testing.assert_allclose(net, expected_equity, rtol=0, atol=1e-6)
    return pd.DataFrame(dict(close=mark, units=units, fee_usd=fee, equity_usd=net)), returns


def statistics(curve, returns, market_open):
    # The opening snapshot anchors the drawdown. Only elapsed hours enter annualization.
    rr = returns.isel(time=slice(1, None))
    equity = qs.calc_equity(returns)
    underwater = qs.calc_underwater(equity)
    points = qs.calc_avg_points_per_year(rr)
    weight = (curve.units * curve.close / curve.equity_usd).to_numpy()[:, None]
    weights = xr.DataArray(weight, dims=("time", "asset"),
                           coords={"time": returns.time, "asset": ["BTC"]})
    opens = weights.copy(data=market_open.to_numpy()[:, None]).expand_dims(field=["open"])
    kwargs = dict(points_per_year=points, min_periods=1)
    return pd.DataFrame({k: v.to_series() for k, v in {
        "equity": equity, "relative_return": rr,
        "volatility": qs.calc_volatility_annualized(rr, **kwargs),
        "underwater": underwater, "max_drawdown": qs.calc_max_drawdown(underwater),
        "sharpe_ratio": qs.calc_sharpe_ratio_annualized(rr, **kwargs),
        "mean_return": qs.calc_mean_return_annualized(rr, **kwargs),
        "bias": qs.calc_bias(weights), "instruments": qs.calc_instruments(weights),
        "avg_turnover": qs.calc_avg_turnover(weights, equity, opens, max_periods=len(weights), **kwargs),
        "avg_holding_time": qs.calc_avg_holding_time(weights, **kwargs),
    }.items()})


def main():
    manifest = json.loads((ARCHIVE / "manifest.json").read_text())
    capital = manifest["run_config"]["initial_capital"]
    market = pd.read_parquet(ROOT / "data/BTC_hourly_2024_2025.parquet")
    orders = replay_orders(market, manifest["windows"])
    index = pd.date_range(manifest["evaluation_start"], manifest["evaluation_end"], freq=HOUR)
    bars = market.set_axis(market.index + HOUR).reindex(index)
    curve, rr = simulate(bars.close.ffill(), orders, capital)
    # Independently reconcile every position and gross mark with the original experiment.
    archived = pd.read_csv(ARCHIVE / "original_equity.csv", index_col=0, parse_dates=True)
    pd.testing.assert_index_equal(archived.index, curve.index, check_names=False)
    np.testing.assert_array_equal(curve.units, archived.units)
    np.testing.assert_allclose(curve.equity_usd + curve.fee_usd.cumsum(), archived.equity,
                               atol=1e-6, rtol=0)
    stats = statistics(curve, rr, bars.open.fillna(curve.close))
    annual = []
    for year in range(index[0].year, (index[-1]-HOUR).year+1):
        part = curve.loc[pd.Timestamp(year, 1, 1):pd.Timestamp(year+1, 1, 1)]
        initial, final = part.equity_usd.iloc[[0, -1]]
        fees = part.fee_usd.iloc[1:].sum()
        annual.append(dict(year=year, initial_usd=initial, final_usd=final, fees_usd=fees,
                           net_return=final/initial-1))
    annual = pd.DataFrame(annual)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    annual.to_csv(OUTPUT / "anual.csv", index=False, float_format="%.10f")
    stats.iloc[[-1]].to_csv(OUTPUT / "kpi_qnt.csv", index_label="time_utc")
    curve.join(stats).to_parquet(OUTPUT / "historico_horario.parquet")
    orders.assign(fee_usd=FEE).to_csv(OUTPUT / "ordenes.csv", index_label="time_utc")
    print(stats.iloc[-1].to_string())
    print(annual.to_string(index=False))
    print(f"Entradas: {(orders.change == 1).sum()}, salidas: {(orders.change == -1).sum()}, "
          f"comisiones: {curve.fee_usd.sum():.2f} USD")


if __name__ == "__main__":
    main()
