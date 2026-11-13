from numbers import Integral
import talib
import pandas as pd
from typing import Dict, Callable
from tester import get_vector_buys
from read_data import ohlc_form, execution_prices
import keys


avalible_methods: set = {"SMA", "EMA", "WMA", "DEMA", "TEMA", "TRIMA", "KAMA", "T3", "MIDPOINT"}


def main(method: str, data, lookback: int, candle: int = 1, shorts: bool = False, asset: str = None, temporality: str = "auto", origin="epoch") -> pd.DataFrame:
    return signals_from(method, ohlc_form(data, candle, temporality, asset, origin), lookback, shorts)


def generate(method: str, lookback: int, candle: int = 1, shorts: bool = False) -> pd.DataFrame:
    return signals_from(method, keys.ohlc_cache[candle], lookback, shorts)


def signals_from(method: str, df: pd.DataFrame, lookback: int, shorts: bool = False) -> pd.DataFrame:
    if isinstance(lookback, bool) or not isinstance(lookback, Integral) or lookback < 2:
        raise ValueError("El período de la media debe ser un entero mayor o igual a 2")
    if method not in avalible_methods:
        raise ValueError("Método no disponible")
    close = df["close"]
    ma = SIMPLE_METHODS[method](close, lookback) if len(df) else close
    signals = get_vector_buys(ma, close, shorts, df.get("is_liquid"))
    return pd.DataFrame({"Signals": signals, "Prices": execution_prices(signals, df)})


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
