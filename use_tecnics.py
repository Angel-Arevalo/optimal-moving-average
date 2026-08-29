import talib
import pandas as pd
import numpy as np
from typing import Dict, Callable
from tester import get_vector_buys

import keys

avalible_methods: set = {"SMA", "EMA", "WMA", "DEMA", "TEMA", "TRIMA", "KAMA", "T3", "MIDPOINT"}

# actualmente este método va a retornar el vector de compras y ventas
# De ahora en adelanta se asume que data ya es el vector de información
# final
def main(method: str, data: pd.Series, lookback: int, candle: int, nooh_data: pd.DataFrame, shorts: bool = False, bid_df: pd.DataFrame = None, ask_df: pd.DataFrame = None) -> pd.DataFrame:

    if method not in avalible_methods:
        raise ValueError("Not avalible method")

    ma: pd.Series = SIMPLE_METHODS[method](data, lookback)
    ma = get_vector_buys(ma, data, nooh_data, shorts)

    if bid_df is not None and ask_df is not None:
        ask_close = ask_df["close"]
        bid_close = bid_df["close"]
    else:
        ask_close = keys.ask_cache[candle]["close"]
        bid_close = keys.bid_cache[candle]["close"]

    precios = np.where(ma == 1, ask_close.loc[ma.index], bid_close.loc[ma.index])

    ma = pd.DataFrame({
        "Signals": ma,
        "Prices": precios
    }, index=ma.index)

    return ma
# Esta función me permite guardar una correspondencia entre
# Strings y funciones de TAB-Lib 

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
