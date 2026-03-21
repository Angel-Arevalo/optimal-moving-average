import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

import read_data
from use_tecnics import SIMPLE_METHODS
from tester import get_vector_buys


def calcular_rsi(precios: pd.Series, periodos: int = 14) -> pd.Series:
    delta = precios.diff()
    ganancia = delta.where(delta > 0, 0.0).ewm(alpha=1/periodos, adjust=False).mean()
    perdida = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/periodos, adjust=False).mean()
    rs = ganancia / perdida
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def calcular_atr(high: pd.Series, low: pd.Series, close: pd.Series, periodos: int = 14) -> pd.Series:
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/periodos, adjust=False).mean()
    return atr.bfill() # Llenar vacíos iniciales

def extraer_matriz_multicanal(v_p: np.ndarray, v_m: np.ndarray, v_rsi: np.ndarray, v_atr: np.ndarray) -> np.ndarray:
    # 1. Distancia
    distancia = (v_p - v_m) / v_m
    
    # 2. Curvatura
    primera_derivada = np.gradient(v_m)
    curvatura = np.gradient(primera_derivada)
    std_curv = np.std(curvatura)
    if std_curv != 0:
        curvatura = (curvatura - np.mean(curvatura)) / std_curv

    # 3. Fourier Suavizado
    fft_vals = np.fft.fft(distancia)
    corte = int(len(fft_vals) * 0.25)
    fft_filtrado = np.copy(fft_vals)
    fft_filtrado[corte:-corte] = 0 
    fft_suavizado = np.real(np.fft.ifft(fft_filtrado))

    # 4. RSI Normalizado (De 0-100 a rango -1 a 1)
    rsi_norm = (v_rsi / 50.0) - 1.0

    # 5. ATR Normalizado (Volatilidad relativa a la ventana actual)
    atr_mean = np.mean(v_atr)
    atr_norm = v_atr / atr_mean if atr_mean != 0 else v_atr

    return np.column_stack((distancia, curvatura, fft_suavizado, rsi_norm, atr_norm))

def procesar_estrategia_con_cnn(data: pd.DataFrame, ma_week: list, window_size=60) -> pd.DataFrame:
    model = tf.keras.models.load_model("modelo_con_5_dimensiones.keras")
    vector_total = []

    for week in ma_week:
        inicio_w = pd.to_datetime(week[0])
        fin_w = pd.to_datetime(week[1])
        metodo = week[2][0]
        candle = week[2][1]
        params = week[2][2]

        time_col = inicio_w - pd.Timedelta(weeks=2)
        data_week = data.loc[time_col: fin_w]

        df_ohlc = read_data.ohlc_form(data_week, f"{candle}min")
        prices = df_ohlc["close"]
        highs = df_ohlc["high"]
        lows = df_ohlc["low"]
        
        ma = SIMPLE_METHODS[metodo](prices, params)
        rsi_full = calcular_rsi(prices)
        atr_full = calcular_atr(highs, lows, prices)

        base_signals_full = get_vector_buys(ma, prices)
        base_signals_week = base_signals_full.loc[inicio_w: fin_w]
        
        if base_signals_week.empty:
            continue
            
        buy_indices = base_signals_week[base_signals_week == 1].index
        sell_indices = base_signals_week[base_signals_week == -1].index
        
        modified_signals = pd.Series(0, index=prices.loc[inicio_w: fin_w].index)
        
        if not sell_indices.empty:
            modified_signals.loc[sell_indices] = -1

        batch_matrices = []
        valid_buys_info = []

        for inicio_compra in buy_indices:
            pos = prices.index.get_loc(inicio_compra)
            if pos >= window_size:
                v_p = prices.iloc[pos - window_size : pos].values
                v_m = ma.iloc[pos - window_size : pos].values
                v_rsi = rsi_full.iloc[pos - window_size : pos].values
                v_atr = atr_full.iloc[pos - window_size : pos].values
                
                matriz_5d = extraer_matriz_multicanal(v_p, v_m, v_rsi, v_atr)
                batch_matrices.append(matriz_5d)
                valid_buys_info.append((inicio_compra, pos))

        if not batch_matrices:
            continue

        X_batch = np.array(batch_matrices)
        
        pred_logs = model.predict(X_batch, verbose=0)
        esperas_velas = np.maximum(0, np.round(np.expm1(pred_logs).flatten())).astype(int)

        for (inicio_compra, pos), espera in zip(valid_buys_info, esperas_velas):
            nueva_pos = pos + espera

            if nueva_pos >= len(prices):
                continue 
                
            idx_objetivo = prices.index[nueva_pos]

            ventas_intermedias = sell_indices[(sell_indices > inicio_compra) & (sell_indices <= idx_objetivo)]
            
            if not ventas_intermedias.empty:
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
    
    for row in df_final.itertuples():
        sig = row.Signals
        if sig == 1 and not trade_abierto:
            clean_rows.append((row.Index, sig, row.Prices))
            trade_abierto = True
        elif sig == -1 and trade_abierto:
            clean_rows.append((row.Index, sig, row.Prices))
            trade_abierto = False
            
    df_limpio = pd.DataFrame(clean_rows, columns=["Time", "Signals", "Prices"]).set_index("Time")
    
    return df_limpio

def get_features_cnn(data: pd.DataFrame, ma_week: list, window_size: int = 60) -> Tuple[np.ndarray, np.ndarray]:
    X_sequences = []
    y_targets = []

    for week in ma_week:
        time_col = week[0] - pd.Timedelta(hours=5) 
        data_week = data.loc[time_col: week[1]]
        
        df_ohlc = read_data.ohlc_form(data_week, f"{week[2][1]}min")
        prices = df_ohlc["close"]
        highs = df_ohlc["high"]
        lows = df_ohlc["low"]
        
        ma = SIMPLE_METHODS[week[2][0]](prices, week[2][2])
        rsi_full = calcular_rsi(prices)
        atr_full = calcular_atr(highs, lows, prices)
        
        signals = get_vector_buys(ma, prices)[week[0]: week[1]]

        if signals.empty: continue

        buy_indices = signals[signals == 1].index
        for inicio in buy_indices:
            try:
                pos_inicio = prices.index.get_loc(inicio)
                if pos_inicio >= window_size:
                    v_p = prices.iloc[pos_inicio - window_size : pos_inicio].values
                    v_m = ma.iloc[pos_inicio - window_size : pos_inicio].values
                    v_rsi = rsi_full.iloc[pos_inicio - window_size : pos_inicio].values
                    v_atr = atr_full.iloc[pos_inicio - window_size : pos_inicio].values

                    matriz_5d = extraer_matriz_multicanal(v_p, v_m, v_rsi, v_atr)
                    
                    ventas = signals[(signals == -1) & (signals.index > inicio)]
                    if not ventas.empty:
                        tramo = prices.loc[inicio : ventas.index[0]]
                        X_sequences.append(matriz_5d)
                        y_targets.append(tramo.argmin())
            except: continue

    return np.array(X_sequences), np.array(y_targets)

def residual_block(x, filters, kernel_size=3):
    shortcut = x
    x = layers.Conv1D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Conv1D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

def build_resnet_trading(input_shape=(60, 5)):
    inputs = layers.Input(shape=input_shape)
    
    x = layers.Conv1D(64, kernel_size=7, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    
    res_x = x
    x = layers.Conv1D(128, kernel_size=3, padding='same', strides=2)(x)
    shortcut = layers.Conv1D(128, kernel_size=1, strides=2)(res_x) 
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    
    x = layers.GlobalAveragePooling1D()(x)
    
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(1)(x)
    
    model = models.Model(inputs, outputs)
    model.compile(optimizer=optimizers.Adam(learning_rate=0.0003), loss=tf.keras.losses.LogCosh())
    
    return model

def train_cnn_velas_model(X: np.ndarray, y: np.ndarray):
    limite_velas = 120 

    validos = ~np.isnan(X).any(axis=(1, 2)) & ~np.isnan(y) & (y < limite_velas)
    X, y = X[validos], y[validos]

    y_log = np.log1p(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.15, shuffle=True)

    model = build_resnet_trading(input_shape=(X.shape[1], X.shape[2]))
    
    lr_reducer = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=8, min_lr=1e-7)
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    print(f"Entrenando ResNet Trading (5 Dimensiones) con {len(X_train)} ejemplos...")
    model.fit(
        X_train, y_train, 
        epochs=250, 
        batch_size=128,
        validation_split=0.1, 
        callbacks=[lr_reducer, early_stop],
        verbose=1
    )

    preds = np.expm1(model.predict(X_test)).flatten()
    y_real = np.expm1(y_test)
    
    mae = mean_absolute_error(y_real, np.round(preds))
    print(f"\nMAE Final (Multicanal 5D): {mae:.4f} velas.")

    return model, mae