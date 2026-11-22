import talib
import pandas as pd
import numpy as np
from typing import Dict, Callable
from tester import get_vector_buys


# Medias que sólo usan precio.
avalible_methods: set = {"SMA", "EMA", "WMA", "DEMA", "TEMA", "TRIMA", "KAMA", "T3", "MIDPOINT", "HMA", "ZLEMA", "ALMA"}

# Medias que además necesitan volumen. Sólo entran a la optimización cuando el
# cache trae volumen (ver ma_methods() y keys.has_volume).
volume_methods: set = {"VWMA"}


def ma_methods(has_volume: bool = True) -> set:
    """Métodos disponibles; excluye los que piden volumen si no lo hay."""
    if has_volume:
        return avalible_methods | volume_methods
    return set(avalible_methods)

# actualmente este método va a retornar el vector de compras y ventas
# De ahora en adelanta se asume que data ya es el vector de información
# final
def main(method: str, data: pd.Series, lookback: int, candle: int, nooh_data: pd.DataFrame, shorts: bool = False, bid_df: pd.DataFrame = None, ask_df: pd.DataFrame = None, daily: bool = False, *, weekly_closes: bool = True, side: str = None) -> pd.DataFrame:

    if method in volume_methods:
        volume = _volume_for(candle, data.index, bid_df, ask_df)
        ma: pd.Series = VOLUME_METHODS[method](data, volume, lookback)
    elif method in avalible_methods:
        ma = SIMPLE_METHODS[method](data, lookback)
    else:
        raise ValueError("Not avalible method")

    ma = get_vector_buys(ma, data, nooh_data, shorts, daily, weekly_closes=weekly_closes, side=side)

    if bid_df is not None and ask_df is not None:
        ask_close = ask_df["close"]
        bid_close = bid_df["close"]
    else:
        import keys
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


# Medias que usan volumen: firma (close, volume, lookback).
VOLUME_METHODS: Dict[str, Callable[[pd.Series, pd.Series, int], pd.Series]] = {}
def _volume_method(name: str) -> Callable:
    def decorador(func: Callable) -> Callable:
        VOLUME_METHODS[name] = func
        return func
    return decorador


def _volume_for(candle: int, index: pd.Index, bid_df: pd.DataFrame = None, ask_df: pd.DataFrame = None) -> pd.Series:
    """Volumen de la vela `candle`, alineado al índice del close de la media."""
    from read_data import VOLUME
    if bid_df is not None and VOLUME in bid_df:
        source = bid_df[VOLUME]
    else:
        import keys
        mid = keys.mid_cache[candle]
        if VOLUME not in mid:
            raise ValueError("El cache no tiene volumen para este método")
        source = mid[VOLUME]
    return source.reindex(index)
    

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

@_all_methods("HMA")
def hma(prices: pd.Series, lookback: int) -> pd.Series:
    return pd.Series(talib.HMA(prices.to_numpy(float), timeperiod=lookback), index=prices.index)

@_all_methods("ZLEMA")
def zlema(prices: pd.Series, lookback: int) -> pd.Series:
    return pd.Series(talib.ZLEMA(prices.to_numpy(float), timeperiod=lookback), index=prices.index)

@_all_methods("ALMA")
def alma(prices: pd.Series, lookback: int) -> pd.Series:
    return pd.Series(talib.ALMA(prices.to_numpy(float), timeperiod=lookback), index=prices.index)

@_volume_method("VWMA")
def vwma(prices: pd.Series, volume: pd.Series, lookback: int) -> pd.Series:
    return pd.Series(
        talib.VWMA(prices.to_numpy(float), volume.to_numpy(float), timeperiod=lookback),
        index=prices.index,
    )
