from typing import Dict, Callable, Optional
import pandas as pd
import numpy as np
import talib
import keys
from tester_dir import backtest
from read_data import execution_prices


def dir_main(signals: pd.DataFrame, data: pd.DataFrame, params: dict, short: bool = False, filter: bool = False):
    rev, tend = split_signals(signals, data, params, filter)
    return backtest(rev, tend, data, short)


def split_signals(signals: pd.DataFrame, data: pd.DataFrame, params: dict, filter: bool = False, condition=None) -> tuple:
    if condition is None:
        method = DIR_METHODS[params["name"]]
        condition = method(signals, params)
    rev, tend = _split_signals_and_change(signals, condition)
    tend["Prices"] = execution_prices(tend["Signals"], data)
    return rev, tend.iloc[:0] if filter else tend


def _split_signals_and_change(signals: pd.DataFrame, condition: pd.Series) -> tuple:
    if signals.empty:
        return signals.copy(), signals.copy()
    entries = signals.index[::2]
    flip = condition.reindex(entries, method="ffill").fillna(False).to_numpy(dtype=bool)
    mask = np.repeat(flip, 2)
    rev, tend = signals.loc[~mask].copy(), signals.loc[mask].copy()
    tend["Signals"] *= -1
    return rev, tend


DIR_METHODS: Dict[str, Callable[[dict], pd.Series]] = {}
def _dir_methods(name: str) -> Callable[[Callable[[dict], pd.Series]], Callable[[dict], pd.Series]]:
    def decorador(func: Callable[[dict], pd.Series]) -> Callable[[dict], pd.Series]:
        DIR_METHODS[name] = func
        return func

    return decorador

@_dir_methods("KEF")
def kaufman(df_signals: pd.DataFrame, params: dict, data: Optional[pd.DataFrame] = None, raw: bool = False) -> pd.Series:
    candle = params["candle"]
    window = params["window"]
    follow_tend = params["follow_tend"]

    cache = data if data is not None else keys.ohlc_cache[candle]
    mid_close: pd.Series = cache["close"]

    change: pd.Series = (mid_close - mid_close.shift(window)).abs()
    single_step_change = (mid_close - mid_close.shift(1)).abs()
    divisor = single_step_change.rolling(window=window).sum()

    kef: pd.Series = (change / divisor).replace([np.inf, -np.inf], np.nan).fillna(0)
    return kef if raw else kef > follow_tend


@_dir_methods("HURST")
def H(df_signals: pd.DataFrame, params: dict, data: Optional[pd.DataFrame] = None, raw: bool = False) -> pd.Series:
    candle = params["candle"]
    window = params["window"]
    follow_tend = params["follow_tend"]

    cache = data if data is not None else keys.ohlc_cache[candle]
    mid_close: pd.Series = cache["close"]
    values = mid_close.to_numpy(dtype=np.float64)
    n_obs = values.shape[0]

    result = np.full(n_obs, 0.5, dtype=np.float64)

    if n_obs >= window and window > 2:
        price_windows = np.lib.stride_tricks.sliding_window_view(values, window)
        returns = np.diff(price_windows, axis=1)

        std = returns.std(axis=1, ddof=1)
        mean_adj = returns - returns.mean(axis=1, keepdims=True)
        cum_dev = np.cumsum(mean_adj, axis=1)

        r = cum_dev.max(axis=1) - cum_dev.min(axis=1)

        with np.errstate(divide="ignore", invalid="ignore"):
            rs = np.where(std > 0, r / std, 0.0)
            n = returns.shape[1]
            hurst = np.log(rs) / np.log(n)

        valid = (std > 0) & (rs > 0) & np.isfinite(hurst)
        hurst_vals = np.where(valid, hurst, 0.5)

        result[window - 1:] = hurst_vals

    hurst_series = pd.Series(result, index=mid_close.index)
    return hurst_series if raw else hurst_series > follow_tend


@_dir_methods("LO_MACKINLAY")
def lo_mackinlay(df_signals: pd.DataFrame, params: dict, data: Optional[pd.DataFrame] = None, raw: bool = False) -> pd.Series:
    candle = params["candle"]
    window = params["window"]
    k = params["k"]
    follow_tend = params["follow_tend"]

    cache = data if data is not None else keys.ohlc_cache[candle]
    mid_close: pd.Series = cache["close"]

    diff_1 = mid_close.diff(1)
    diff_k = mid_close.diff(k)

    var_1 = diff_1.rolling(window=window).var()
    var_k = diff_k.rolling(window=window).var()

    vr = (var_k / (k * var_1)).replace([np.inf, -np.inf], np.nan)
    return vr if raw else vr > follow_tend


@_dir_methods("ADX")
def adx(df_signals: pd.DataFrame, params: dict, data: Optional[pd.DataFrame] = None, raw: bool = False) -> pd.Series:
    candle = params["candle"]
    window = params["window"]
    follow_tend = params["follow_tend"]

    cache = data if data is not None else keys.ohlc_cache[candle]
    high: pd.Series = cache["high"]
    low: pd.Series = cache["low"]
    close: pd.Series = cache["close"]

    adx_series = pd.Series(talib.ADX(high.to_numpy(float), low.to_numpy(float),
                                    close.to_numpy(float), timeperiod=window), index=close.index).fillna(0)

    return adx_series if raw else adx_series > follow_tend


@_dir_methods("VOL_RATIO")
def volatility_ratio(df_signals: pd.DataFrame, params: dict, data: Optional[pd.DataFrame] = None, raw: bool = False) -> pd.Series:
    candle = params["candle"]
    window = params["window"]
    follow_tend = params["follow_tend"]

    cache = data if data is not None else keys.ohlc_cache[candle]
    high: pd.Series = cache["high"]
    low: pd.Series = cache["low"]
    close: pd.Series = cache["close"]

    sma = close.rolling(window=window).mean()
    std = close.rolling(window=window).std()
    bbw = (4 * std) / sma

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = np.fmax(np.fmax(tr1, tr2), tr3)
    atr_pct = tr.rolling(window=window).mean() / sma

    v_ratio = (bbw / (2 * atr_pct)).replace([np.inf, -np.inf], np.nan).fillna(0)
    return v_ratio if raw else v_ratio > follow_tend


@_dir_methods("SHANNON")
def shannon_entropy(df_signals: pd.DataFrame, params: dict, data: Optional[pd.DataFrame] = None, raw: bool = False) -> pd.Series:
    candle = params["candle"]
    window = params["window"]
    bins = params["bins"]
    follow_tend = params["follow_tend"]

    cache = data if data is not None else keys.ohlc_cache[candle]
    mid_close: pd.Series = cache["close"]
    values = mid_close.to_numpy(dtype=np.float64)
    n_obs = values.shape[0]

    result = np.full(n_obs, 1.0, dtype=np.float64)
    max_ent = np.log2(bins)

    if n_obs >= window:
        price_windows = np.lib.stride_tricks.sliding_window_view(values, window)
        returns = np.diff(price_windows, axis=1)
        M, W = returns.shape

        r_min = returns.min(axis=1, keepdims=True)
        r_max = returns.max(axis=1, keepdims=True)
        r_range = (r_max - r_min).ravel()

        all_zero = r_range == 0

        safe_range = np.where(r_range == 0, 1.0, r_range)[:, None]
        bin_idx = np.floor((returns - r_min) / safe_range * bins).astype(np.int64)
        bin_idx = np.clip(bin_idx, 0, bins - 1)

        row_idx = np.repeat(np.arange(M), W)
        flat_idx = row_idx * bins + bin_idx.ravel()
        counts = np.bincount(flat_idx, minlength=M * bins).reshape(M, bins)

        probs = counts / counts.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            ent_terms = np.where(probs > 0, probs * np.log2(probs), 0.0)
        ent = -ent_terms.sum(axis=1)

        entropy_vals = np.where(max_ent > 0, ent / max_ent, 1.0) if max_ent > 0 else np.ones(M)
        entropy_vals = np.where(all_zero, 1.0, entropy_vals)

        result[window - 1:] = entropy_vals

    entropy = pd.Series(result, index=mid_close.index)
    return entropy if raw else entropy < follow_tend


@_dir_methods("ATR_EXPANSION")
def atr_expansion(df_signals: pd.DataFrame, params: dict, data: Optional[pd.DataFrame] = None, raw: bool = False) -> pd.Series:
    candle = params["candle"]
    window = params["window"]
    lookback_ma = params["lookback_ma"]
    follow_tend = params["follow_tend"]

    cache = data if data is not None else keys.ohlc_cache[candle]
    high: pd.Series = cache["high"]
    low: pd.Series = cache["low"]
    close: pd.Series = cache["close"]

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = np.fmax(np.fmax(tr1, tr2), tr3)

    atr = tr.rolling(window=window).mean()
    atr_ma = atr.rolling(window=lookback_ma).mean()

    atr_ratio = (atr / atr_ma).replace([np.inf, -np.inf], np.nan).fillna(0)
    return atr_ratio if raw else atr_ratio > follow_tend


@_dir_methods("KELTNER_BREAKOUT")
def keltner_breakout(df_signals: pd.DataFrame, params: dict, data: Optional[pd.DataFrame] = None, raw: bool = False) -> pd.Series:
    candle = params["candle"]
    window = params["window"]
    mult = params["mult"]

    cache = data if data is not None else keys.ohlc_cache[candle]
    high: pd.Series = cache["high"]
    low: pd.Series = cache["low"]
    close: pd.Series = cache["close"]

    sma = close.rolling(window=window).mean()

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = np.fmax(np.fmax(tr1, tr2), tr3)
    atr = tr.rolling(window=window).mean()

    upper = sma + mult * atr
    lower = sma - mult * atr

    is_breakout = (close > upper) | (close < lower)
    return is_breakout.fillna(False)
