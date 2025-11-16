import talib
import pandas as pd 
from typing import Dict, Callable


simple_methods: set = {"EMA", "DEMA", "SMA"}
complex_methods: set = set()
avalible_methods: set = simple_methods | complex_methods

def main(method: str, lookback: int, data: pd.DataFrame) -> pd.Series:
    if method not in avalible_methods:
        raise ValueError("Not avalible method")


    if method in simple_methods:
        return SIMPLE_METHODS[method](data["close"], lookback)
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

@_all_methods("DEMA")
def dema(prices: pd.Series, lookback: int) -> pd.Series:
    return pd.Series(talib.DEMA(prices.to_numpy(float), timeperiod=lookback), index=prices.index)
