from typing import Tuple, Dict, Callable, Optional
import pandas as pd
from skopt.space import Real, Integer, Categorical

import numpy as np
import keys
from tester_dir import hit_ratio, risk_reward, profit_factor, mae, sqn


def dir_main(signals_and_prices: pd.DataFrame, data: pd.DataFrame, candle_ma: int, params_method: dict, short: bool) -> Tuple[float, float, float, float, float]:
    cond_vector: pd.Series = DIR_METHODS[params_method["name"]](signals_and_prices, params_method)

    rever_tr, trend_tr = _split_signals_and_change(signals_and_prices, cond_vector, short, data)

    hr = hit_ratio(rever_tr, trend_tr, short)
    rr = risk_reward(rever_tr, trend_tr, short)
    pr = profit_factor(rever_tr, trend_tr, short)
    tr = len(rever_tr)//2 + len(trend_tr)//2
    mae_val = mae(rever_tr, trend_tr, short, candle_ma)
    return hr, rr, pr, tr, mae_val, sqn(rever_tr, trend_tr, short)

def _split_signals_and_change(signals_and_prices: pd.DataFrame, change_dir_cond: pd.Series, short: bool, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:

    entry_sig: int = -1 if short else 1

    entrys = signals_and_prices[signals_and_prices["Signals"] == entry_sig]

    if entrys.empty:
        return None, None

    entry_idx = entrys.index.to_numpy()
    cond_idx = change_dir_cond.index.to_numpy()

    pos = np.searchsorted(cond_idx, entry_idx, side="right") - 2
    valid_mask = pos >= 0

    cond_values = change_dir_cond.values
    trade_flip_cond = np.where(valid_mask, cond_values[pos], False)

    full_flip_mask = np.repeat(trade_flip_cond, 2)

    df_reversion = signals_and_prices[~full_flip_mask].copy()

    df_trend = signals_and_prices[full_flip_mask].copy()
    df_trend["Signals"] = df_trend["Signals"] * -1

    filter_data = data.reindex(df_trend.index, method="bfill")
    df_trend["Prices"] = np.where(df_trend["Signals"] == 1, filter_data["ask"], filter_data["bid"])

    return df_reversion, df_trend

DIR_METHODS: Dict[str, Callable[[dict], pd.Series]] = {}
def _dir_methods(name: str) -> Callable[[Callable[[dict], pd.Series]], Callable[[dict], pd.Series]]:
    def decorador(func: Callable[[dict], pd.Series]) -> Callable[[dict], pd.Series]:
        DIR_METHODS[name] = func
        return func

    return decorador

@_dir_methods("KEF")
def kaufman(df_signals: pd.DataFrame, params: dict, data: Optional[pd.DataFrame] = None) -> pd.Series:
    candle = params["candle"]
    window = params["window"]
    follow_tend = params["follow_tend"]

    cache = data if data is not None else keys.mid_cache[candle]
    mid_close: pd.Series = cache["close"]

    change: pd.Series = (mid_close - mid_close.shift(window)).abs()
    single_step_change = (mid_close - mid_close.shift(1)).abs()
    divisor = single_step_change.rolling(window=window).sum()

    kef: pd.Series = (change / divisor).replace([np.inf, -np.inf], np.nan).fillna(0)
    return kef > follow_tend


@_dir_methods("HURST")
def H(df_signals: pd.DataFrame, params: dict, data: Optional[pd.DataFrame] = None) -> pd.Series:
    candle = params["candle"]
    window = params["window"]
    follow_tend = params["follow_tend"]

    cache = data if data is not None else keys.mid_cache[candle]
    mid_close: pd.Series = cache["close"]
    values = mid_close.to_numpy(dtype=np.float64)
    n_obs = values.shape[0]

    result = np.full(n_obs, 0.5, dtype=np.float64)

    if n_obs >= window:
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
    return hurst_series > follow_tend


@_dir_methods("LO_MACKINLAY")
def lo_mackinlay(df_signals: pd.DataFrame, params: dict, data: Optional[pd.DataFrame] = None) -> pd.Series:
    candle = params["candle"]
    window = params["window"]
    k = params["k"]
    follow_tend = params["follow_tend"]

    cache = data if data is not None else keys.mid_cache[candle]
    mid_close: pd.Series = cache["close"]

    diff_1 = mid_close.diff(1)
    diff_k = mid_close.diff(k)

    var_1 = diff_1.rolling(window=window).var()
    var_k = diff_k.rolling(window=window).var()

    vr = (var_k / (k * var_1)).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    return vr > follow_tend


@_dir_methods("ADX")
def adx(df_signals: pd.DataFrame, params: dict, data: Optional[pd.DataFrame] = None) -> pd.Series:
    candle = params["candle"]
    window = params["window"]
    follow_tend = params["follow_tend"]

    cache = data if data is not None else keys.mid_cache[candle]
    high: pd.Series = cache["high"]
    low: pd.Series = cache["low"]
    close: pd.Series = cache["close"]

    up_move = high.diff(1)
    down_move = -low.diff(1)

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = np.fmax(np.fmax(tr1, tr2), tr3)

    atr_val = tr.ewm(span=window, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=close.index).ewm(span=window, adjust=False).mean() / atr_val
    minus_di = 100 * pd.Series(minus_dm, index=close.index).ewm(span=window, adjust=False).mean() / atr_val

    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di)).replace([np.inf, -np.inf], np.nan)
    adx_series = dx.ewm(span=window, adjust=False).mean().fillna(0)

    return adx_series > follow_tend


@_dir_methods("VOL_RATIO")
def volatility_ratio(df_signals: pd.DataFrame, params: dict, data: Optional[pd.DataFrame] = None) -> pd.Series:
    candle = params["candle"]
    window = params["window"]
    follow_tend = params["follow_tend"]

    cache = data if data is not None else keys.mid_cache[candle]
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
    return v_ratio > follow_tend


@_dir_methods("SHANNON")
def shannon_entropy(df_signals: pd.DataFrame, params: dict, data: Optional[pd.DataFrame] = None) -> pd.Series:
    candle = params["candle"]
    window = params["window"]
    bins = params["bins"]
    follow_tend = params["follow_tend"]

    cache = data if data is not None else keys.mid_cache[candle]
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
    return entropy < follow_tend


@_dir_methods("ATR_EXPANSION")
def atr_expansion(df_signals: pd.DataFrame, params: dict, data: Optional[pd.DataFrame] = None) -> pd.Series:
    candle = params["candle"]
    window = params["window"]
    lookback_ma = params["lookback_ma"]
    follow_tend = params["follow_tend"]

    cache = data if data is not None else keys.mid_cache[candle]
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
    return atr_ratio > follow_tend


@_dir_methods("KELTNER_BREAKOUT")
def keltner_breakout(df_signals: pd.DataFrame, params: dict, data: Optional[pd.DataFrame] = None) -> pd.Series:
    candle = params["candle"]
    window = params["window"]
    mult = params["mult"]

    cache = data if data is not None else keys.mid_cache[candle]
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
