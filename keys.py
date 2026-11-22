from use_tecnics import avalible_methods
import read_data
import pandas as pd
#Temporalidad de particionado
temp: str = "min"
side: str = "revert"
# llaves para las llamadas del optimizador
calls: int = 60
initial_points: int = 20


# llaves (básicas) para el espacio de búsqueda
lookbacks: int = 110
lookbacks_min: int = 2

candles: int = 120
candles_min: int = 1

# Filtro opcional de métodos. None = sin filtro (usa todos los disponibles,
# incluidos los de volumen si el cache lo trae). Un conjunto/tupla restringe.
methods = None

# pre-calculo de ohlc
bid_cache: dict = {}
ask_cache: dict = {}
mid_cache: dict = {}

# ¿Las velas del cache traen columna de volumen? Lo fija fill_ohlc_dict.
# Los métodos que necesitan volumen no entran a la optimización si es False.
has_volume: bool = False

def cache_has_volume() -> bool:
    from read_data import VOLUME
    return any(VOLUME in mid for mid in mid_cache.values())

def fill_ohlc_dict(data) -> bool:
    global has_volume
    bid_cache.clear()
    ask_cache.clear()
    mid_cache.clear()
    data_ = read_data.prepare_asset(data)
    source = None
    if "bid" not in data_ and temp in ["min", "h"]:
        source = read_data.ohlc_resolution(data_.index)
    for i in range(1, candles + 1):
        if source is not None:
            duration = pd.Timedelta(i, unit=temp)
            if duration < source or duration % source:
                continue
        bid, ask, mid = read_data.prepared_ohlc(data_, i, temp)
        if not mid.empty:
            bid_cache[i], ask_cache[i], mid_cache[i] = bid, ask, mid
    if not any(i >= candles_min for i in mid_cache):
        raise ValueError("No hay velas compatibles con los datos en el rango de keys.temp y keys.candles")
    has_volume = cache_has_volume()
    return temp == "d"
