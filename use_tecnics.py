import talib
import pandas as pd 
from typing import Dict, Callable, Union
from tester import get_vector_buys

simple_methods: set = {"SMA", "EMA", "WMA", "DEMA", "TEMA", "TRIMA", "KAMA", "T3", "MIDPOINT"}
complex_methods: set = {"MACD", "BBANDS", "DONCHIAN", "ZSCORE_EMA"} 
avalible_methods: set = simple_methods | complex_methods


# actualmente este método va a retornar el vector de compras y ventas
def main(method: str, data: pd.DataFrame, adicional_data: Union[list, int]) -> pd.DataFrame:
    if method not in avalible_methods:
        raise ValueError("Not avalible method")

    if isinstance(adicional_data, int) and method in complex_methods:
        raise ValueError("Todo método simple usa solo lookback")

    if method in simple_methods:
        if isinstance(adicional_data, list):
            adicional_data = adicional_data[0]

        ma: pd.Series = SIMPLE_METHODS[method](data["close"], adicional_data)
        ma = get_vector_buys(ma, data["close"])

    else:
        if method == "MACD":
            # en adicional_data se espera así
            # slowperiod, fastperiod, signal_back 
            ma: pd.Series = macd(data["close"], 
                                adicional_data[0], 
                                adicional_data[1],
                                adicional_data[2])

        elif method == "BBANDS":
            # en adicional_data se espera así
            # lookback, dev_up, dev_dn, matype
            ma: pd.Series = bbands(data["close"],
                                   adicional_data[0],
                                   adicional_data[1],
                                   adicional_data[2],
                                   adicional_data[3])

        elif method == "DONCHIAN":
            ma: pd.Series = donchian(data["close"], 
                                     adicional_data[0], 
                                     adicional_data[1])

        elif method == "ZSCORE_EMA":
            ma: pd.Series = z_score_ema(data["close"],
                                        adicional_data[0],
                                        adicional_data[1],
                                        adicional_data[2])

        else:
            raise ValueError(f"{method} no implementado")

    ma = pd.concat([ma, data["close"]], axis = 1, join = "inner")
    ma.columns = ["Signals", "Prices"]
    return ma

# Esta función me permite guardar una correspondencia entre
# Strings y funciones de TAB-Lib 
#
# La idea es que se usa el decorador @_all_methods("str") y justo debajo
# una función que retorne la función deseada.
SIMPLE_METHODS: Dict[str, Callable[[pd.Series, int], pd.Series]] = {}
# No es posible dar una forma general para llamar elementos de COMPLEX_METHODS
# pues cada uno puede tener una forma de llamarse
COMPLEX_METHODS: Dict[str, Callable[[pd.Series], pd.Series]] = {}
def _all_methods(name: str) -> Callable[[pd.Series, int], pd.Series]:
    def decorador(func: Callable[[pd.Series, int], pd.Series]) -> Callable[[pd.Series, int], pd.Series]:
        if name in simple_methods:
            SIMPLE_METHODS[name] = func
            return func

        COMPLEX_METHODS[name] = func
        return func

    return decorador
    

@_all_methods("SMA")
def sma(prices: pd.Series, lookback: int) -> pd.Series:
    return pd.Series(talib.SMA(prices.to_numpy(dtype=float), timeperiod=lookback), index=prices.index)

@_all_methods("EMA")
def ema(prices: pd.Series, lookback: int) -> pd.Series:
    return pd.Series(talib.EMA(prices.to_numpy(float), timeperiod=lookback), index=prices.index)

@_all_methods("WMA")
def wma(prices: pd.Series, lookback: int) -> pd.Series:
    return pd.Series(talib.WMA(prices.to_numpy(float), timeperiod=lookback), index=prices.index)

@_all_methods("DEMA")
def dema(prices: pd.Series, lookback: int) -> pd.Series:
    return pd.Series(talib.DEMA(prices.to_numpy(float), timeperiod=lookback), index=prices.index)

@_all_methods("TEMA")
def tema(prices: pd.Series, lookback: int) -> pd.Series:
    return pd.Series(talib.TEMA(prices.to_numpy(float), timeperiod=lookback), index=prices.index)

@_all_methods("TRIMA")
def trima(prices: pd.Series, lookback: int) -> pd.Series:
    return pd.Series(talib.TRIMA(prices.to_numpy(float), timeperiod=lookback), index=prices.index)

@_all_methods("KAMA")
def kama(prices: pd.Series, lookback: int) -> pd.Series:
    return pd.Series(talib.KAMA(prices.to_numpy(float), timeperiod=lookback), index=prices.index)

@_all_methods("T3")
def T3(prices: pd.Series, lookback: int) -> pd.Series:
    return pd.Series(talib.T3(prices.to_numpy(float), timeperiod=lookback), index=prices.index)

@_all_methods("MIDPOINT")
def midpoint(prices: pd.Series, lookback: int) -> pd.Series:
    return pd.Series(talib.MIDPOINT(prices.to_numpy(float), timeperiod=lookback), index=prices.index)

# conjunto de funciones complejas
@_all_methods("MACD")
def macd(prices: pd.Series, lookback_max: int, lookback_min: int, signal_back: int) -> pd.Series:
    macd, macdsignal, _ = talib.MACD(
                       prices.to_numpy(float), 
                       fastperiod=lookback_min, 
                       slowperiod=lookback_max, 
                       signalperiod=signal_back
                     )

    macd = pd.Series(macd, index=prices.index)
    macdsignal = pd.Series(macdsignal, index=prices.index)

    return get_vector_buys(macdsignal, macd)

@_all_methods("BBANDS")
def bbands(prices: pd.Series, lookback: int, dev_up: float, dev_dn: float, matype: int) -> pd.Series:
    upper, middle, lower = talib.BBANDS(
        prices.to_numpy(float), 
        timeperiod=lookback, 
        nbdevup=dev_up, 
        nbdevdn=dev_dn,
        matype=matype
    )

    upper = pd.Series(upper, index=prices.index)
    lower = pd.Series(lower, index=prices.index)

    pre_prices = prices.shift(1)
    pre_lower = lower.shift(1)
    pre_upper = upper.shift(1)

    signal_buy = ((pre_prices > pre_lower) & (prices <= lower)).astype(int)

    signal_sell = ((pre_prices < pre_upper) & (prices >= upper)).astype(int)

    vector_signals = signal_buy - signal_sell
    return vector_signals[vector_signals != 0]

@_all_methods("DONCHIAN")
def donchian(prices: pd.Series, lookback_upper: int, lookback_lower: int) -> pd.Series:
    if lookback_lower is None:
        lookback_lower = lookback_upper

    upper = prices.shift(1).rolling(window=lookback_upper).max()
    lower = prices.shift(1).rolling(window=lookback_lower).min()

    signal_buy = (prices > upper).astype(int)
    signal_sell = (prices < lower).astype(int)

    vector_signals = signal_buy - signal_sell
    vector_signals = vector_signals[vector_signals != 0]
    return vector_signals[vector_signals != vector_signals.shift(1)]

@_all_methods("ZSCORE_EMA")
def z_score_ema(prices: pd.Series, lookback: int, threshold: float, matype: int) -> pd.Series:
    ma = talib.MA(prices.to_numpy(float), timeperiod=lookback, matype=matype)
    std = prices.rolling(window=lookback).std()

    z_score = pd.Series((prices - ma) / std, index=prices.index)
    pre_z = z_score.shift(1)

    if threshold == 0.0:
        signal_buy = ((pre_z < 0) & (z_score >= 0)).astype(int)
        signal_sell = ((pre_z > 0) & (z_score <= 0)).astype(int)
    else:
        signal_buy = ((pre_z < -threshold) & (z_score >= -threshold)).astype(int)
        signal_sell = ((pre_z > threshold) & (z_score <= threshold)).astype(int)

    vector_signals = signal_buy - signal_sell
    return vector_signals[vector_signals != 0]
