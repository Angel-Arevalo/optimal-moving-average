from numbers import Integral
import pandas as pd
import read_data


temp: str = "auto"
origin: str = "epoch"
calls: int = 60
initial_points: int = 20
lookbacks: int = 110
lookbacks_min: int = 2
candles: int = 1
candles_min: int = 1
methods: set[str] = {"SMA", "EMA", "WMA", "DEMA", "TEMA", "TRIMA", "KAMA", "T3", "MIDPOINT"}
ohlc_cache: dict = {}


def validate_space() -> None:
    for value in (lookbacks_min, lookbacks, candles_min, candles):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError("Los límites de búsqueda deben ser enteros")
    if lookbacks_min < 2 or lookbacks < lookbacks_min:
        raise ValueError("Espacio de búsqueda inválido para lookback")
    if candles_min < 1 or candles < candles_min:
        raise ValueError("Espacio de búsqueda inválido para sesiones")
    from use_tecnics import avalible_methods
    if not methods or not set(methods).issubset(avalible_methods):
        raise ValueError("Seleccione medias móviles disponibles")


def fill_ohlc_dict(data, asset: str = None, requested=None) -> pd.DataFrame:
    validate_space()
    df = read_data.read_asset(data, asset)
    ohlc_cache.clear()
    for candle in sorted(set(range(candles_min, candles + 1) if requested is None else requested)):
        ohlc_cache[candle] = read_data.resample_asset(df, candle, temp, origin)
    return df
