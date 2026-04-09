from use_tecnics import avalible_methods
import read_data
from typing import Union
# llaves para las llamadas del optimizador
calls: int = 60
initial_points: int = 20


# llaves (básicas) para el espacio de búsqueda
lookbacks: int = 110
candles: int = 100
n_rsis: int = 50
methods: set[str] = avalible_methods

# pre-calculo de ohlc
pre_ohlc: dict = {}

def fill_ohlc_dict(data, is_bid = False) -> None:
    pre_ohlc.clear()

    if not is_bid and isinstance(data, str):
        data = read_data.read_asset(data)

    for i in range(1, candles+1):
        pre_ohlc[i] = read_data.ohlc_form(data, i, is_bid)

        if not is_bid:
            pre_ohlc[i] = pre_ohlc[i]["close"]
