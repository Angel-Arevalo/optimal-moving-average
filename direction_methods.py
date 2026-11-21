from typing import Tuple, Dict, Callable, Optional, List
import pandas as pd
from skopt.space import Real, Integer, Categorical

import numpy as np
import talib
import keys
from tester_dir import hit_ratio, risk_reward, profit_factor, mae, sqn
from find_best import f

def dir_main(signals_and_prices: pd.DataFrame, data: pd.DataFrame, candle_ma: int, params_method: dict, short: bool, filter: str, daily: bool = False) -> Tuple[float, ...]:
    cond_vector: pd.Series = DIR_METHODS[params_method["name"]](signals_and_prices, params_method)

    rever_tr, trend_tr = _split_signals_and_change(signals_and_prices, cond_vector, short, data, candle_ma, params_method["candle"], daily=daily)

    tr_filtered = len(trend_tr)//2
    empty_df = pd.DataFrame({"Signals": [], "Prices": []})

    hr_re, hr_td = hit_ratio(rever_tr, empty_df, short), hit_ratio(empty_df, trend_tr, not short)
    rr_re, rr_td = risk_reward(rever_tr, empty_df, short), risk_reward(empty_df, trend_tr, not short)
    pr_re, pr_td = profit_factor(rever_tr, empty_df, short), profit_factor(empty_df, trend_tr, not short)
    tr_re, tr_td = len(rever_tr)//2, tr_filtered
    sq_re, sq_td = sqn(rever_tr, empty_df, short), sqn(empty_df, trend_tr, not short)
    ma_re, ma_td = mae(rever_tr, empty_df, short, candle_ma), mae(empty_df, trend_tr, not short, candle_ma)

    f_re = f(hr_re,
             rr_re,
             pr_re,
             tr_re,
             sq_re,
             ma_re
    )

    f_td = f(hr_td,
             rr_td,
             pr_td,
             tr_td,
             sq_td,
             ma_td
    )


    if filter == "both":
        pass
    elif filter == "revert":
        trend_tr =  empty_df
    elif filter == "tend":
        rever_tr = empty_df
    else:
        raise ValueError("No definido")

    hr = hit_ratio(rever_tr, trend_tr, short)
    rr = risk_reward(rever_tr, trend_tr, short)
    pr = profit_factor(rever_tr, trend_tr, short)
    tr = len(rever_tr)//2 + len(trend_tr)//2
    mae_val = mae(rever_tr, trend_tr, short, candle_ma)

    return hr, rr, pr, tr, mae_val, sqn(rever_tr, trend_tr, short), tr_filtered, f_re, f_td

def trend_at_entries(closed_condition: pd.Series, entry_times: pd.DatetimeIndex) -> pd.Series:
    """Last detector close available at each entry; default to reversion before the first."""
    return closed_condition.reindex(entry_times, method="ffill", fill_value=False)


def _split_signals_and_change(signals_and_prices: pd.DataFrame, change_dir_cond: pd.Series, short: bool, data: pd.DataFrame, 
                              candle_ma: int, candle_dir: int, bid_df: pd.DataFrame = None, ask_df: pd.DataFrame = None, daily: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:

    entry_sig: int = -1 if short else 1

    if bid_df is not None and ask_df is not None:
        ask_close = ask_df["close"]
        bid_close = bid_df["close"]
    else:
        ask_close = keys.ask_cache[candle_ma]["close"]
        bid_close = keys.bid_cache[candle_ma]["close"]


    entrys = signals_and_prices[signals_and_prices["Signals"] == entry_sig]

    # En diario los dos índices identifican la sesión ya cerrada.
    entrada = entrys.index
    cierres_detector = change_dir_cond.index
    if not daily and keys.temp != "d":
        entrada = entrada + pd.Timedelta(candle_ma, unit=keys.temp)
        cierres_detector = cierres_detector + pd.Timedelta(candle_dir, unit=keys.temp)
    closed_condition = change_dir_cond.set_axis(cierres_detector)
    mask_completa = np.repeat(trend_at_entries(closed_condition, entrada).to_numpy(), 2)

    df_trend: pd.DataFrame = signals_and_prices[mask_completa].copy()
    df_reversion: pd.DataFrame = signals_and_prices[~mask_completa].copy()

    if not df_trend.empty:
        df_trend["Signals"] = -df_trend["Signals"]

        df_trend["Prices"] = np.where(
            df_trend["Signals"] == 1, 
            ask_close.loc[df_trend.index], 
            bid_close.loc[df_trend.index]
        )

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



@_dir_methods("OU_REVERSION")
def ou_reversion(df_signals: pd.DataFrame, params: dict, data: Optional[pd.DataFrame] = None) -> pd.Series:
    candle = params["candle"]
    window = params["window"]
    follow_tend = params["follow_tend"]

    cache = data if data is not None else keys.mid_cache[candle]
    mid_close: pd.Series = cache["close"]
    values = mid_close.to_numpy(dtype=np.float64)
    n_obs = values.shape[0]

    theta_result = np.zeros(n_obs, dtype=np.float64)

    if n_obs >= window:
        price_windows = np.lib.stride_tricks.sliding_window_view(values, window)

        x_prev = price_windows[:, :-1]
        x_curr = price_windows[:, 1:]

        n = x_prev.shape[1]

        mean_prev = x_prev.mean(axis=1, keepdims=True)
        mean_curr = x_curr.mean(axis=1, keepdims=True)

        cov = ((x_prev - mean_prev) * (x_curr - mean_curr)).sum(axis=1)
        var_prev = ((x_prev - mean_prev) ** 2).sum(axis=1)

        with np.errstate(divide="ignore", invalid="ignore"):
            b = np.where(var_prev > 0, cov / var_prev, 1.0)

        b_safe = np.clip(b, 1e-6, 0.999999)

        theta = -np.log(b_safe)

        theta = np.where(b < 0.999999, theta, 0.0)

        theta = np.where(b > 1e-6, theta, 0.0)

        theta_result[window - 1:] = theta

    theta_series = pd.Series(theta_result, index=mid_close.index)

    return theta_series < follow_tend


# --- Detectores de régimen basados en volumen ---------------------------------
# Todos leen cache["volume"], que keys.fill_ohlc_dict conserva sólo cuando el
# origen trae volumen. find_best_dir los excluye de la optimización si no lo hay
# (ver keys.has_volume). El guard evita un KeyError si se invocan sin volumen.

# Detectores de dirección que requieren volumen.
VOLUME_DIR_METHODS = {"OBV_SLOPE", "MFI", "ADOSC", "VWAP_DIST", "MARKETFI", "VOL_SURGE"}


def _require_volume(cache) -> pd.Series:
    if "volume" not in cache:
        raise ValueError("Este detector de dirección requiere volumen y el cache no lo tiene")
    return cache["volume"]


@_dir_methods("OBV_SLOPE")
def obv_slope(df_signals: pd.DataFrame, params: dict, data: Optional[pd.DataFrame] = None) -> pd.Series:
    """Pendiente del OBV normalizada: volumen que respalda el movimiento = tendencia."""
    candle = params["candle"]
    window = params["window"]
    follow_tend = params["follow_tend"]

    cache = data if data is not None else keys.mid_cache[candle]
    close: pd.Series = cache["close"]
    volume = _require_volume(cache)

    obv = pd.Series(talib.OBV(close.to_numpy(float), volume.to_numpy(float)), index=close.index)
    # Cambio relativo del OBV en la ventana, escalado por el volumen medio para
    # hacerlo comparable entre activos y tramos.
    obv_change = obv - obv.shift(window)
    vol_scale = volume.rolling(window=window).sum().replace(0, np.nan)
    slope = (obv_change / vol_scale).replace([np.inf, -np.inf], np.nan).fillna(0)

    return slope.abs() > follow_tend


@_dir_methods("MFI")
def money_flow_index(df_signals: pd.DataFrame, params: dict, data: Optional[pd.DataFrame] = None) -> pd.Series:
    """Money Flow Index: lejos del centro (50) = presión direccional = tendencia."""
    candle = params["candle"]
    window = params["window"]
    follow_tend = params["follow_tend"]

    cache = data if data is not None else keys.mid_cache[candle]
    high: pd.Series = cache["high"]
    low: pd.Series = cache["low"]
    close: pd.Series = cache["close"]
    volume = _require_volume(cache)

    mfi = pd.Series(
        talib.MFI(high.to_numpy(float), low.to_numpy(float), close.to_numpy(float),
                  volume.to_numpy(float), timeperiod=window),
        index=close.index,
    ).fillna(50.0)

    # Distancia al centro 50 (0 a 50). follow_tend es el umbral de esa distancia.
    return (mfi - 50.0).abs() > follow_tend


@_dir_methods("ADOSC")
def ad_oscillator(df_signals: pd.DataFrame, params: dict, data: Optional[pd.DataFrame] = None) -> pd.Series:
    """Oscilador de acumulación/distribución: magnitud alta = flujo con dirección."""
    candle = params["candle"]
    fastperiod = params["fastperiod"]
    slowperiod = params["slowperiod"]
    follow_tend = params["follow_tend"]

    cache = data if data is not None else keys.mid_cache[candle]
    high: pd.Series = cache["high"]
    low: pd.Series = cache["low"]
    close: pd.Series = cache["close"]
    volume = _require_volume(cache)

    # ADOSC sólo tiene sentido con slow > fast; forzarlo evita osciladores degenerados.
    if slowperiod <= fastperiod:
        slowperiod = fastperiod + 1

    adosc = pd.Series(
        talib.ADOSC(high.to_numpy(float), low.to_numpy(float), close.to_numpy(float),
                    volume.to_numpy(float), fastperiod=fastperiod, slowperiod=slowperiod),
        index=close.index,
    ).fillna(0.0)

    # Normalizar por el volumen medio para que follow_tend sea comparable.
    vol_scale = volume.rolling(window=slowperiod).mean().replace(0, np.nan)
    norm = (adosc / vol_scale).replace([np.inf, -np.inf], np.nan).fillna(0)

    return norm.abs() > follow_tend


@_dir_methods("VWAP_DIST")
def vwap_distance(df_signals: pd.DataFrame, params: dict, data: Optional[pd.DataFrame] = None) -> pd.Series:
    """Distancia relativa del precio al VWAP rolling: lejos del VWAP = tendencia."""
    candle = params["candle"]
    window = params["window"]
    follow_tend = params["follow_tend"]

    cache = data if data is not None else keys.mid_cache[candle]
    high: pd.Series = cache["high"]
    low: pd.Series = cache["low"]
    close: pd.Series = cache["close"]
    volume = _require_volume(cache)

    typical = (high + low + close) / 3.0
    pv = (typical * volume).rolling(window=window).sum()
    vol_sum = volume.rolling(window=window).sum().replace(0, np.nan)
    vwap = pv / vol_sum

    dist = ((close - vwap).abs() / vwap).replace([np.inf, -np.inf], np.nan).fillna(0)

    return dist > follow_tend


@_dir_methods("MARKETFI")
def market_facilitation(df_signals: pd.DataFrame, params: dict, data: Optional[pd.DataFrame] = None) -> pd.Series:
    """Market Facilitation Index (Williams): rango por unidad de volumen.

    Un MARKETFI alto respecto a su media = el movimiento avanza con poco volumen
    relativo (eficiente) = tendencia.
    """
    candle = params["candle"]
    window = params["window"]
    follow_tend = params["follow_tend"]

    cache = data if data is not None else keys.mid_cache[candle]
    high: pd.Series = cache["high"]
    low: pd.Series = cache["low"]
    volume = _require_volume(cache)

    mktfi = pd.Series(
        talib.MARKETFI(high.to_numpy(float), low.to_numpy(float), volume.to_numpy(float)),
        index=high.index,
    ).replace([np.inf, -np.inf], np.nan)

    mktfi_ma = mktfi.rolling(window=window).mean()
    ratio = (mktfi / mktfi_ma).replace([np.inf, -np.inf], np.nan).fillna(0)

    return ratio > follow_tend


@_dir_methods("VOL_SURGE")
def volume_surge(df_signals: pd.DataFrame, params: dict, data: Optional[pd.DataFrame] = None) -> pd.Series:
    """Volumen actual vs. su media: expansión de volumen acompaña rupturas = tendencia."""
    candle = params["candle"]
    window = params["window"]
    follow_tend = params["follow_tend"]

    cache = data if data is not None else keys.mid_cache[candle]
    volume = _require_volume(cache)

    vol_ma = volume.rolling(window=window).mean().replace(0, np.nan)
    surge = (volume / vol_ma).replace([np.inf, -np.inf], np.nan).fillna(0)

    return surge > follow_tend


# --- Detectores de régimen por geometría, ciclo y momentum ---------------------
# Sólo usan precio (high/low/close); no requieren volumen.

@_dir_methods("LINREG_R2")
def linreg_r2(df_signals: pd.DataFrame, params: dict, data: Optional[pd.DataFrame] = None) -> pd.Series:
    """R² de la regresión del precio contra el tiempo: recta limpia = tendencia.

    R² alto (cercano a 1) = el precio sigue una línea; bajo = lateral/ruidoso.
    Se calcula como el cuadrado de la correlación precio-tiempo en la ventana.
    """
    candle = params["candle"]
    window = params["window"]
    follow_tend = params["follow_tend"]

    cache = data if data is not None else keys.mid_cache[candle]
    close: pd.Series = cache["close"]

    # Serie temporal lineal (0,1,2,...) como variable independiente.
    t = pd.Series(np.arange(len(close), dtype=np.float64), index=close.index)
    corr = pd.Series(
        talib.CORREL(close.to_numpy(float), t.to_numpy(float), timeperiod=window),
        index=close.index,
    )
    r2 = (corr ** 2).replace([np.inf, -np.inf], np.nan).fillna(0)

    return r2 > follow_tend


@_dir_methods("LINREG_SLOPE")
def linreg_slope(df_signals: pd.DataFrame, params: dict, data: Optional[pd.DataFrame] = None) -> pd.Series:
    """Pendiente de la regresión lineal, normalizada por precio: empinada = tendencia."""
    candle = params["candle"]
    window = params["window"]
    follow_tend = params["follow_tend"]

    cache = data if data is not None else keys.mid_cache[candle]
    close: pd.Series = cache["close"]

    slope = pd.Series(
        talib.LINEARREG_SLOPE(close.to_numpy(float), timeperiod=window),
        index=close.index,
    )
    # Normalizar por el precio hace el umbral comparable entre activos.
    norm_slope = (slope / close).replace([np.inf, -np.inf], np.nan).fillna(0)

    return norm_slope.abs() > follow_tend


@_dir_methods("HT_TRENDMODE")
def ht_trendmode(df_signals: pd.DataFrame, params: dict, data: Optional[pd.DataFrame] = None) -> pd.Series:
    """Hilbert Transform Trend vs Cycle Mode: clasificador nativo de TA-Lib.

    Devuelve 1 en modo tendencia, 0 en modo ciclo. No usa ventana ni umbral.
    """
    candle = params["candle"]

    cache = data if data is not None else keys.mid_cache[candle]
    close: pd.Series = cache["close"]

    mode = pd.Series(
        talib.HT_TRENDMODE(close.to_numpy(float)),
        index=close.index,
    ).fillna(0)

    return mode > 0


@_dir_methods("AROONOSC")
def aroon_oscillator(df_signals: pd.DataFrame, params: dict, data: Optional[pd.DataFrame] = None) -> pd.Series:
    """Aroon Oscillator (-100 a 100): magnitud alta = tendencia establecida."""
    candle = params["candle"]
    window = params["window"]
    follow_tend = params["follow_tend"]

    cache = data if data is not None else keys.mid_cache[candle]
    high: pd.Series = cache["high"]
    low: pd.Series = cache["low"]

    osc = pd.Series(
        talib.AROONOSC(high.to_numpy(float), low.to_numpy(float), timeperiod=window),
        index=high.index,
    ).fillna(0)

    return osc.abs() > follow_tend


@_dir_methods("DI_BALANCE")
def di_balance(df_signals: pd.DataFrame, params: dict, data: Optional[pd.DataFrame] = None) -> pd.Series:
    """Diferencia |+DI - -DI|: sesgo direccional neto (no sólo la fuerza como ADX)."""
    candle = params["candle"]
    window = params["window"]
    follow_tend = params["follow_tend"]

    cache = data if data is not None else keys.mid_cache[candle]
    high: pd.Series = cache["high"]
    low: pd.Series = cache["low"]
    close: pd.Series = cache["close"]

    plus_di = pd.Series(
        talib.PLUS_DI(high.to_numpy(float), low.to_numpy(float), close.to_numpy(float), timeperiod=window),
        index=close.index,
    ).fillna(0)
    minus_di = pd.Series(
        talib.MINUS_DI(high.to_numpy(float), low.to_numpy(float), close.to_numpy(float), timeperiod=window),
        index=close.index,
    ).fillna(0)

    return (plus_di - minus_di).abs() > follow_tend


# --- Detector probabilístico de deriva (camino más probable) -------------------

@_dir_methods("DRIFT_PROB")
def drift_prob(df_signals: pd.DataFrame, params: dict, data: Optional[pd.DataFrame] = None) -> pd.Series:
    """Camino más probable bajo un random walk con deriva (Monte Carlo analítico).

    Se estiman deriva (mu) y volatilidad (sigma) de los log-retornos en la ventana.
    Bajo un paseo aleatorio con deriva, la probabilidad de que el precio termine
    por encima del actual tras `horizon` pasos es Phi(mu*sqrt(h)/sigma). Esto es el
    límite exacto al que converge un Monte Carlo de muchos caminos, sin ruido ni
    coste de simular.

    Se mide el SESGO del camino: |p - 0.5|. Un sesgo alto (los caminos se van
    mayoritariamente hacia un lado) indica deriva dominante = régimen tendencial;
    un sesgo cercano a 0 (caminos repartidos) = sin dirección probable = reversión.
    Igual que los demás detectores, devuelve True en tendencia; el motor decide a
    favor o en contra al invertir la señal de la media base.
    """
    candle = params["candle"]
    window = params["window"]
    horizon = params.get("horizon", 1)
    follow_tend = params["follow_tend"]

    cache = data if data is not None else keys.mid_cache[candle]
    close: pd.Series = cache["close"]
    values = close.to_numpy(dtype=np.float64)
    n_obs = values.shape[0]

    prob_bias = np.zeros(n_obs, dtype=np.float64)

    # Log-retornos; se necesita una ventana de retornos para mu y sigma.
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ret = np.diff(np.log(np.where(values > 0, values, np.nan)))

    if log_ret.shape[0] >= window:
        ret_windows = np.lib.stride_tricks.sliding_window_view(log_ret, window)
        mu = np.nanmean(ret_windows, axis=1)
        sigma = np.nanstd(ret_windows, axis=1, ddof=1)

        # z = mu*sqrt(h)/sigma  ->  p_up = Phi(z);  sesgo = |p_up - 0.5|
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(sigma > 0, mu * np.sqrt(horizon) / sigma, 0.0)
        # Phi(z) con la relación Phi(z) = 0.5*(1 + erf(z/sqrt(2))).
        p_up = 0.5 * (1.0 + _erf(z / np.sqrt(2.0)))
        bias = np.abs(p_up - 0.5)
        bias = np.where(np.isfinite(bias), bias, 0.0)

        start = window  # primer índice de close con ventana completa de retornos
        prob_bias[start:start + bias.shape[0]] = bias

    bias_series = pd.Series(prob_bias, index=close.index)
    return bias_series > follow_tend


def _erf(x: np.ndarray) -> np.ndarray:
    try:
        from scipy.special import erf as _scipy_erf
        return _scipy_erf(x)
    except Exception:

        sign = np.sign(x)
        x = np.abs(x)
        t = 1.0 / (1.0 + 0.3275911 * x)
        y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                    - 0.284496736) * t + 0.254829592) * t * np.exp(-x * x)
        return sign * y
