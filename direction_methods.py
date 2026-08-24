from typing import Tuple, Dict, Callable
import pandas as pd

import numpy as np
import keys

def dir_main(signals_and_prices: pd.DataFrame, noohlc_data: pd.DataFrame, params_method: list) -> Tuple[pd.DataFrame, pd.DataFrame]:

    return (None, None)

def _split_signals_and_change(signals_and_prices: pd.DataFrame, change_dir_cond: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:

    reutnr (None, None)

DIR_METHODS: Dict[str, Callable] = {}
def _dir_methods(name: str) -> Callable[[pd.Series, int], pd.Series]:
    def decorador(func: Callable[[pd.Series, int], pd.Series]) -> Callable[[pd.Series, int], pd.Series]:
        DIR_METHODS[name] = func
        return func

    return decorador

dir_methods: set[str] = DIR_METHODS
@_dir_methods("KEF")
def kaufman(candle: int, window: int, follow_tend: float = .5) -> pd.Series:
    mid_close: pd.Series = keys.mid_cache[candle]["close"]

    change: pd.Series = (mid_close - mid_close.shift(window)).abs()

    single_step_change = (mid_close - mid_close.shift(1)).abs()
    divisor = single_step_change.rolling(window=window).sum()

    kef: pd.Series = (change / divisor).replace([np.inf, -np.inf], np.nan).fillna(0)
    # Condición de cambio de dirección
    return (kef > follow_tend)

@_dir_methods("HURST")
def H(candle: int, window: int, follow_tend: float = 0.55) -> pd.Series:
    mid_close: pd.Series = keys.mid_cache[candle]["close"]
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
def lo_mackinlay(candle: int, window: int, k: int = 4, follow_tend: float = 1.2) -> pd.Series:
    mid_close: pd.Series = keys.mid_cache[candle]["close"]

    diff_1 = mid_close.diff(1)
    diff_k = mid_close.diff(k)

    var_1 = diff_1.rolling(window=window).var()
    var_k = diff_k.rolling(window=window).var()

    vr = (var_k / (k * var_1)).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    return vr > follow_tend

@_dir_methods("ADX")
def adx(candle: int, window: int = 14, follow_tend: float = 25.0) -> pd.Series:
    high: pd.Series = keys.mid_cache[candle]["high"]
    low: pd.Series = keys.mid_cache[candle]["low"]
    close: pd.Series = keys.mid_cache[candle]["close"]

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
def volatility_ratio(candle: int, window: int = 20, follow_tend: float = 1.2) -> pd.Series:
    high: pd.Series = keys.mid_cache[candle]["high"]
    low: pd.Series = keys.mid_cache[candle]["low"]
    close: pd.Series = keys.mid_cache[candle]["close"]

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
def shannon_entropy(candle: int, window: int = 30, bins: int = 10, follow_tend: float = 0.85) -> pd.Series:
    mid_close: pd.Series = keys.mid_cache[candle]["close"]
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
def atr_expansion(candle: int, window: int = 14, lookback_ma: int = 50, follow_tend: float = 1.3) -> pd.Series:
    high: pd.Series = keys.mid_cache[candle]["high"]
    low: pd.Series = keys.mid_cache[candle]["low"]
    close: pd.Series = keys.mid_cache[candle]["close"]

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = np.fmax(np.fmax(tr1, tr2), tr3)

    atr = tr.rolling(window=window).mean()
    atr_ma = atr.rolling(window=lookback_ma).mean()

    atr_ratio = (atr / atr_ma).replace([np.inf, -np.inf], np.nan).fillna(0)
    return atr_ratio > follow_tend

@_dir_methods("KELTNER_BREAKOUT")
def keltner_breakout(candle: int, window: int = 20, mult: float = 2.0) -> pd.Series:
    high: pd.Series = keys.mid_cache[candle]["high"]
    low: pd.Series = keys.mid_cache[candle]["low"]
    close: pd.Series = keys.mid_cache[candle]["close"]

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
