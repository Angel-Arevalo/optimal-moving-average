from typing import NamedTuple
import numpy as np
import pandas as pd


class Metrics(NamedTuple):
    hr: float
    rr: float
    pr: float
    tr: int
    mae: float
    sqn: float


def trade_returns(signals: pd.DataFrame, shorts: bool = False) -> pd.Series:
    validate_signals(signals, shorts)
    entries = signals["Prices"].iloc[::2].to_numpy(dtype=float)
    exits = signals["Prices"].iloc[1::2].to_numpy(dtype=float)
    if len(entries) != len(exits):
        raise ValueError("Cada entrada debe tener una salida")
    values = (exits / entries - 1) * (-1 if shorts else 1)
    return pd.Series(values, index=signals.index[1::2], dtype=float)


def summarize(trades: pd.Series, mae_val: float = 0.0) -> Metrics:
    if not np.isfinite(trades.to_numpy(dtype=float)).all() or not np.isfinite(mae_val) or mae_val < 0:
        raise ValueError("Los rendimientos y el MAE deben ser finitos; el MAE no puede ser negativo")
    sigma = trades.std()
    sqn = np.sqrt(min(len(trades), 100)) * trades.mean() / sigma if sigma > 0 else 0.0
    return Metrics(hit_ratio(trades), rr_ratio(trades), profit_ratio(trades), len(trades), float(mae_val), float(sqn))


def backtest(signals: pd.DataFrame, data: pd.DataFrame, shorts: bool = False) -> Metrics:
    return summarize(trade_returns(signals, shorts), mae(signals, data, shorts))


def get_vector_buys(ma: pd.Series, close: pd.Series, shorts: bool = False, liquid: pd.Series = None) -> pd.Series:
    valid = ma.notna() & ma.shift().notna()
    below = ma > close
    vector = (below.astype(int).diff().fillna(0)).where(valid, 0)
    entry = -1 if shorts else 1
    pending = None
    pairs = []
    for date, signal in vector[vector != 0].items():
        if pending is None and signal == entry:
            if liquid is None or bool(liquid.loc[date]):
                pending = date
        elif pending is not None and signal == -entry:
            pairs.extend([(pending, entry), (date, -entry)])
            pending = None
    dates = pd.DatetimeIndex([date for date, _ in pairs], name=close.index.name)
    return pd.Series([signal for _, signal in pairs], index=dates, dtype=int)


def hit_ratio(trades: pd.Series) -> float:
    return float((trades > 0).mean()) if len(trades) else 0.0


def rr_ratio(trades: pd.Series) -> float:
    win, loss = trades[trades > 0], trades[trades < 0]
    if win.empty:
        return 0.0
    denominator = -loss.mean() if len(loss) else 0.0
    return float(win.mean() / max(denominator, 1e-12))


def profit_ratio(trades: pd.Series) -> float:
    win, loss = trades[trades > 0], trades[trades < 0]
    if win.empty:
        return 0.0
    return float(win.sum() / max(-loss.sum(), 1e-12))


def mae(signals: pd.DataFrame, data: pd.DataFrame, shorts: bool = False) -> float:
    validate_signals(signals, shorts)
    if signals.empty:
        return 0.0
    if (data.empty or not data.index.is_monotonic_increasing or data.index.has_duplicates
            or signals.index[0] < data.index[0] or signals.index[-1] > data.index[-1]):
        raise ValueError("Las cotizaciones deben estar ordenadas y cubrir las operaciones")
    quotes = "ask_close" in data
    field = ("ask_high" if shorts else "bid_low") if quotes else ("high" if shorts else "low")
    extremes = data[field].to_numpy()
    starts = data.index.searchsorted(signals.index[::2], side="right")
    ends = data.index.searchsorted(signals.index[1::2], side="right")
    prices = signals.Prices.iloc[::2].to_numpy()
    marks = data["ask_close" if shorts else "bid_close"].to_numpy() if quotes else None
    values = []
    for start, end, price in zip(starts, ends, prices):
        period = extremes[start:end]
        extreme = (period.max() if shorts else period.min()) if len(period) else price
        if quotes:
            extreme = max(marks[start - 1], extreme) if shorts else min(marks[start - 1], extreme)
        adverse = extreme / price - 1 if shorts else 1 - extreme / price
        values.append(max(0.0, adverse) * 100)
    return float(np.mean(values))


def validate_signals(signals: pd.DataFrame, shorts: bool = False) -> None:
    if len(signals) % 2:
        raise ValueError("Cada entrada debe tener una salida")
    if not isinstance(signals.index, pd.DatetimeIndex) or signals.index.hasnans or signals.index.has_duplicates or not signals.index.is_monotonic_increasing:
        raise ValueError("Las señales deben tener fechas únicas y ordenadas")
    prices = signals["Prices"].to_numpy(dtype=float)
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("Los precios de las señales deben ser positivos y finitos")
    if "Signals" in signals:
        expected = np.tile([-1, 1] if shorts else [1, -1], len(signals) // 2)
        if not np.array_equal(signals.Signals.to_numpy(), expected):
            raise ValueError("La dirección de las señales no coincide con el backtest")
