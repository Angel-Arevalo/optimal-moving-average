import tensorflow as tf
import numpy as np

import pandas as pd
from use_tecnics import main as get_ma_signals

import read_data
from use_tecnics import SIMPLE_METHODS
from tester import get_vector_buys

model = tf.keras.models.load_model("modelo_cnn_velas.keras")

def preparar_ventana_prediccion(prices, ma, window_size=60):
    if len(prices) < window_size:
        return None
        
    v_p = prices.iloc[-window_size:].values
    v_m = ma.iloc[-window_size:].values
    
    secuencia = (v_p - v_m) / v_m
    
    return secuencia.reshape((1, window_size, 1))

def obtener_espera_optima(data_ohlc, metodo, params, window_size=60):
    prices = data_ohlc["close"]

    ma = SIMPLE_METHODS[metodo](prices, params)
    
    X_input = preparar_ventana_prediccion(prices, ma, window_size)
    
    if X_input is not None:
        pred_log = model.predict(X_input, verbose=0)
        
        velas_espera = np.expm1(pred_log)[0][0]
        return int(np.maximum(0, np.round(velas_espera)))
    
    return 0

def procesar_estrategia_con_cnn(data: pd.DataFrame, ma_week: list, window_size=60) -> pd.DataFrame:
    """
    Toma la lista de configuraciones (ma_week), calcula las señales originales,
    predice el retraso para las compras y CANCELA el trade si hay un cruce
    contrario (-1) antes de llegar a la vela objetivo.
    """
    vector_total = []

    for week in ma_week:
        inicio_w = pd.to_datetime(week[0])
        fin_w = pd.to_datetime(week[1])
        metodo = week[2][0]
        candle = week[2][1]
        params = week[2][2]

        time_col = inicio_w - pd.Timedelta(weeks=2)
        data_week = data.loc[time_col: fin_w]

        prices = read_data.ohlc_form(data_week, f"{candle}min")["close"]
        ma = SIMPLE_METHODS[metodo](prices, params)

        base_signals_full = get_vector_buys(ma, prices)
        
        base_signals_week = base_signals_full.loc[inicio_w: fin_w]
        
        if base_signals_week.empty:
            continue
            
        buy_indices = base_signals_week[base_signals_week == 1].index
        sell_indices = base_signals_week[base_signals_week == -1].index
        
        modified_signals = pd.Series(0, index=prices.loc[inicio_w: fin_w].index)
        
        for idx in sell_indices:
            modified_signals.loc[idx] = -1

        for inicio_compra in buy_indices:
            pos = prices.index.get_loc(inicio_compra)

            if pos >= window_size:
                v_p = prices.iloc[pos - window_size : pos].values
                v_m = ma.iloc[pos - window_size : pos].values
                
                secuencia = (v_p - v_m) / v_m
                X_input = secuencia.reshape((1, window_size, 1))
                
                pred_log = model.predict(X_input, verbose=0)
                espera_velas = int(np.maximum(0, np.round(np.expm1(pred_log)[0][0])))
                
                nueva_pos = pos + espera_velas

                if nueva_pos >= len(prices):
                    continue 
                    
                idx_objetivo = prices.index[nueva_pos]

                ventas_intermedias = base_signals_full.loc[inicio_compra : idx_objetivo]
                
                if -1 in ventas_intermedias.values:
                    print(f"Trade abortado: Cruce bajista detectado durante la espera de {espera_velas} velas.")
                    continue 

                if idx_objetivo in modified_signals.index:
                    if modified_signals.loc[idx_objetivo] != -1:
                        modified_signals.loc[idx_objetivo] = 1
            else:
                if inicio_compra in modified_signals.index:
                    modified_signals.loc[inicio_compra] = 1
        df_semana = pd.concat([modified_signals[modified_signals != 0], prices.loc[inicio_w: fin_w]], axis=1, join="inner")
        df_semana.columns = ["Signals", "Prices"]
        vector_total.append(df_semana)

    if not vector_total:
        return pd.DataFrame()
        
    df_final = pd.concat(vector_total)

    clean_rows = []
    trade_abierto = False
    
    for idx, row in df_final.iterrows():
        sig = row["Signals"]
        if sig == 1 and not trade_abierto:
            clean_rows.append((idx, sig, row["Prices"]))
            trade_abierto = True
        elif sig == -1 and trade_abierto:
            clean_rows.append((idx, sig, row["Prices"]))
            trade_abierto = False
            
    df_limpio = pd.DataFrame(clean_rows, columns=["Time", "Signals", "Prices"]).set_index("Time")
    
    return df_limpio