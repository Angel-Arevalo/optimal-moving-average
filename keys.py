from use_tecnics import avalible_methods
import read_data
from typing import Union
#Temporalidad de particionado
temp: str = "min"

# llaves para las llamadas del optimizador
calls: int = 60
initial_points: int = 20


# llaves (básicas) para el espacio de búsqueda
lookbacks: int = 110
lookbacks_min: int = 2

candles: int = 120
candles_min: int = 1

methods: set[str] = avalible_methods

# pre-calculo de ohlc
bid_cache: dict = {}
ask_cache: dict = {}
mid_cache: dict = {}

def fill_ohlc_dict(data) -> None:
    bid_cache.clear()
    ask_cache.clear()
    mid_cache.clear()
    data_ = data.copy()

    for i in range(1, candles+1):
        bid_cache[i], ask_cache[i], mid_cache[i] = read_data.ohlc_form(data_, i, temp)
