"""Reuse original OHLC, MA, optimizer and signal code; expose availability times."""
from contextlib import contextmanager
import numpy as np
import pandas as pd

import keys
from find_best import opti_main
from read_data import prepare_asset, prepared_ohlc, ohlc_resolution
from use_tecnics import SIMPLE_METHODS, main
from .config import FEATURES, FeatureConfig, RunConfig


@contextmanager
def optimizer_settings(config: RunConfig):
    changes = dict(temp=config.temporality, calls=config.optimizer_calls,
                   candles_min=1, candles=config.candle_max, lookbacks_min=2,
                   lookbacks=config.lookback_max, methods=tuple(sorted(config.methods)),
                   bid_cache={}, ask_cache={}, mid_cache={})
    saved = {k: getattr(keys, k) for k in changes}
    try:
        for k, v in changes.items():
            setattr(keys, k, v)
        yield
    finally:
        for k, v in saved.items():
            setattr(keys, k, v)


def optimize(data, config, *, verbose=False):
    with optimizer_settings(config):
        result = opti_main(data, verbose=verbose, engie="gp", shorts=False)
    return str(result[0]), int(result[1]), int(result[2])


def validate_market(data):
    """This experiment accepts a single OHLC asset with bar-start timestamps."""
    if not isinstance(data.index, pd.DatetimeIndex) or not data.index.is_monotonic_increasing:
        raise ValueError("Market index must be ordered datetimes")
    if data.index.has_duplicates or data.index.hasnans or data.index.tz is not None:
        raise ValueError("Use unique UTC-naive timestamps")
    if not set(("open", "high", "low", "close")).issubset(data.columns):
        raise ValueError("This adapter requires OHLC, not tick bid/ask")
    if not np.isfinite(data[["open", "high", "low", "close"]].to_numpy()).all():
        raise ValueError("OHLC must be finite; missing source candles must be omitted")
    if (data[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("Nonpositive OHLC: log features unsafe; refuse this instrument")
    if len(data) < 2:
        raise ValueError("At least two source bars required")


def available_index(index, candle, temporality):
    # Daily labels represent the final session; availability is end of that day.
    return index + (pd.Timedelta(days=1) if temporality == "d"
                    else pd.Timedelta(candle, unit=temporality))


def strategy_view(data, parameters, temporality="h", *, weekly_closes=False):
    """Raw original crossover conditions at availability time, without liquidation.

    weekly_closes is solely a legacy compatibility diagnostic. The optimizer
    still calls its original main(); evaluation/calibration use raw crossings.
    """
    method, candle, lookback = parameters
    bid, ask, mid = prepared_ohlc(prepare_asset(data), candle, temporality)
    ma = SIMPLE_METHODS[method](mid.close, lookback)
    if weekly_closes:
        signals = main(method, mid.close, lookback, candle, data, shorts=False,
                       bid_df=bid, ask_df=ask, daily=temporality == "d")
    else:
        buy = (ma.shift(1) <= mid.close.shift(1)) & (ma > mid.close)
        sell = (ma.shift(1) > mid.close.shift(1)) & (ma <= mid.close)
        crossings = buy.astype(int) - sell.astype(int)
        crossings = crossings.loc[crossings != 0]
        signals = pd.DataFrame({"Signals": crossings,
                                "Prices": np.where(crossings == 1,
                                                   ask.close.loc[crossings.index],
                                                   bid.close.loc[crossings.index])})
    signals = signals.copy()
    signals.index = available_index(signals.index, candle, temporality)
    ma.index = available_index(ma.index, candle, temporality)
    return ma, signals


def build_features(price, ma, config=FeatureConfig()):
    """Both indices are availability times. Past-only MA asof alignment."""
    if not price.index.is_monotonic_increasing or price.index.has_duplicates:
        raise ValueError("Unordered or duplicate observations")
    if not ma.index.is_monotonic_increasing or ma.index.has_duplicates:
        raise ValueError("Unordered or duplicate MA observations")
    aligned = ma.reindex(price.index, method="ffill")
    p = price.where((price > 0) & np.isfinite(price))
    m = aligned.where((aligned > 0) & np.isfinite(aligned))
    lp, lm = np.log(p), np.log(m)
    distance = lp - lm
    n = config.efficiency_period
    denominator = p.diff().abs().rolling(n, min_periods=n).sum()
    er = (p - p.shift(n)).abs() / denominator.where(denominator != 0)
    er = er.mask(denominator.eq(0), 0.)
    result = pd.DataFrame({
        "distance_to_ma": distance,
        "distance_change": distance.diff(),
        "ma_slope": lm.diff(config.slope_period) / config.slope_period,
        "realized_volatility": np.sqrt(lp.diff().pow(2).rolling(
            config.volatility_period, min_periods=config.volatility_period).sum()),
        "efficiency_ratio": er,
    })
    # A nonpositive overshooting DEMA/TEMA invalidates the entire current vector.
    result.loc[p.isna() | m.isna(), :] = np.nan
    return result.loc[:, FEATURES].replace([np.inf, -np.inf], np.nan)


def observations(data, ma, config=FeatureConfig()):
    source = ohlc_resolution(data.index)
    price = data.close.copy()
    price.index = price.index + source
    # Missing source bars create missing observations, never interpolated prices.
    grid = pd.date_range(price.index[0], price.index[-1], freq=source)
    price = price.reindex(grid)
    return price, build_features(price, ma, config)
