import talib
import pandas as pd 
from typing import Dict, Callable

simple_methods: set = {"SMA", "EMA", "WMA", "DEMA", "TEMA", "TRIMA", "KAMA", "T3", "MIDPOINT"}
complex_methods: set = set()
avalible_methods: set = simple_methods | complex_methods

def main(method: str, lookback: int, data: pd.DataFrame) -> pd.Series:
    if method not in avalible_methods:
        raise ValueError("Not avalible method")


    if method in simple_methods:
        return SIMPLE_METHODS[method](data["close"], lookback).bfill()
    else:
        raise ValueError("For now only the simple methods are applied")



# Esta función me permite guardar una correspondencia entre
# Strings y funciones de TAB-Lib 
#
# La idea es que se usa el decorador @_all_methods("str") y justo debajo
# una función que retorne la función deseada.
SIMPLE_METHODS: Dict[str, Callable[[pd.Series, int], pd.Series]] = {}
def _all_methods(name: str) -> Callable[[pd.Series, int], pd.Series]:
    def decorador(func: Callable[[pd.Series, int], pd.Series]) -> Callable[[pd.Series, int], pd.Series]:
        SIMPLE_METHODS[name] = func
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

