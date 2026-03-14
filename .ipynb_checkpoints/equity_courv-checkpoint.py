import pandas as pd
import numpy as np
import find_best
from train_network import train_little_model

import keys
from typing import Union, Callable

import read_data
from use_tecnics import main, simple_methods

keys.candles = 20
keys.methods = simple_methods 

ventana_entrenamiento = pd.Timedelta(weeks=3) + pd.Timedelta(days=4)
inicio_testeo = pd.Timedelta(days=3)

fin_testeo = pd.Timedelta(days=4) 

def analice(data: Union[str, pd.DataFrame]) -> None:
    data = fix_data(data)
    initial_mon: float = 1000

    cantidad_comprada: int = 0

    inicio_train = data.index[0].normalize()
    fin_datos = data.index[-1]

    while True:
        train_time = inicio_train + ventana_entrenamiento 
        inicio_test = train_time + inicio_testeo
        fin_test = inicio_test + fin_testeo

        if fin_test >= fin_datos:
            break

        print(f"Entrenamiento: {inicio_train.strftime('%Y-%m-%d')} a {train_time.strftime('%Y-%m-%d')}")

        sub_data = data[inicio_train: train_time]
        score_fft = analizar_regimen_fft(sub_data)

        if score_fft > .2:
            keys.methods = {"ZSCORE_EMA", "BBANDS"}
        else:
            keys.methods = simple_methods | {"MACD", "DONCHIAN"}

        best_ma = find_best.opti_main(sub_data)

        minutos_de_colchon = best_ma[1] * best_ma[2] * (1 if best_ma[0] in simple_methods else 8)
        tiempo_colchon = train_time - pd.Timedelta(minutes=minutos_de_colchon)
 
        bloque_completo = pd.concat([data[tiempo_colchon: train_time - pd.Timedelta(minutes=1)],
                                     data[inicio_test: fin_test]])

        bloque_completo = read_data.ohlc_form(bloque_completo, f"{best_ma[1]}min")

        señales_df = main(best_ma[0], bloque_completo["close"], best_ma[2:])
        test_signals = señales_df.loc[inicio_test : fin_test]

        print(f"Iniciando semana de test ({inicio_test.strftime('%m-%d')} al {fin_test.strftime('%m-%d')}). Liquidez: {initial_mon:.2f}")

        for index in test_signals.index:
            precio_actual = test_signals["Prices"][index]
            signal = test_signals["Signals"][index]

            if signal == 1 and cantidad_comprada == 0:
                cantidad_comprada = int(initial_mon / precio_actual)

                if cantidad_comprada == 0:
                    continue

                precio_compra = cantidad_comprada * precio_actual
                initial_mon -= precio_compra 

            elif signal == -1 and cantidad_comprada != 0:
                cantidad_ganada = precio_actual * cantidad_comprada
                initial_mon += cantidad_ganada
                cantidad_comprada = 0

        inicio_train = inicio_train + pd.Timedelta(days=7)

        print(f"Termina la semana. Liquidez: {initial_mon:.2f}")

def analizar_regimen_fft(precios: pd.Series) -> float:
    precios_centrados = precios - np.mean(precios)
    fft_resultado = np.fft.fft(precios_centrados)

    n = len(precios_centrados)
    energia = np.abs(fft_resultado)[:n // 2]

    energia[0] = 0
    energia_total = np.sum(energia)

    if energia_total == 0:
        return 0.0

    energia_maxima = np.max(energia)

    score_fft = energia_maxima / energia_total

    return score_fft


def fix_data(data: Union[str, pd.DataFrame]) -> pd.DataFrame:
    if isinstance(data, str):
        data = read_data.read_asset(data)

    primer_lunes = data[data.index.dayofweek == 0].index[0]

    return data[primer_lunes:] 
