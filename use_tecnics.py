import talib
import pandas as pd
import numpy as np
from typing import Dict, Callable, Union
from tester import get_vector_buys

avalible_methods: set = {"SMA", "EMA", "WMA", "DEMA", "TEMA", "TRIMA", "KAMA", "T3", "MIDPOINT"}

# actualmente este método va a retornar el vector de compras y ventas
# De ahora en adelanta se asume que data ya es el vector de información
# final
def main(method: str, data: pd.Series, adicional_data: Union[list, int]) -> pd.DataFrame:
    if method not in avalible_methods:
        raise ValueError("Not avalible method")

    if method in avalible_methods:
        if isinstance(adicional_data, list):
            adicional_data = adicional_data[0]

        if len(data.columns) > 1:
            ma: pd.Series = SIMPLE_METHODS[method](data["Precio Spot"], adicional_data)
            ma = get_vector_buys(ma, data["Precio Spot"])
        else:
            ma: pd.Series = SIMPLE_METHODS[method](data, adicional_data)
            ma = get_vector_buys(ma, data)

    else:
        raise ValueError(f"{method} no implementado")

    if len(data.columns) > 1:
        filter = data.loc[ma.index]

        precios = np.where(ma == 1, filter['ask'], filter['bid'])

        ma = pd.DataFrame({
            "Signals": ma,
            "Prices": precios
        }, index=ma.index)

    else:
        ma = pd.concat([ma, data], axis = 1, join = "inner")

    ma.columns = ["Signals", "Prices"]
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
