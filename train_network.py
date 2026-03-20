import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

import read_data
from use_tecnics import main, SIMPLE_METHODS
from tester import get_vector_buys

def get_info_in_week(data: pd.DataFrame, week: list) -> pd.DataFrame:
    time_col = week[0] - pd.Timedelta(weeks=1)
    data_week_raw = data.loc[time_col: week[1]]
    velas_cerradas = read_data.ohlc_form(data_week_raw, f"{week[2][1]}min")["close"]
    method_name = week[2][0]
    params = week[2][2]
    
    df_week = main(method_name, velas_cerradas, params)
    df_week = df_week.loc[week[0]: week[1]]
    
    if len(df_week) == 0:
        return pd.DataFrame()

    f_i = df_week.index[0]
    l_i = df_week.index[-1]
    if df_week["Signals"][f_i] == -1:
        f_i = df_week.index[1] if len(df_week) > 1 else f_i
    if df_week["Signals"][l_i] == 1:
        l_i = df_week.index[-2] if len(df_week) > 1 else l_i

    return df_week.loc[f_i: l_i]

def get_signals_and_prices(data: pd.DataFrame, ma_week: list) -> pd.DataFrame:
    signals_and_prices = None
    for week in ma_week:
        data_week = get_info_in_week(data, week)

        if signals_and_prices is None:
            signals_and_prices = data_week
        else:
            signals_and_prices = pd.concat([signals_and_prices, data_week])
    return signals_and_prices

def get_features_cnn(data: pd.DataFrame, ma_week: list, window_size: int = 60) -> Tuple[np.ndarray, np.ndarray]:
    """Extrae ventanas con una ventana ligeramente mayor para capturar más contexto."""
    X_sequences = []
    y_targets = []

    for week in ma_week:
        time_col = week[0] - pd.Timedelta(weeks=2) # Más historial de colchon
        data_week = data.loc[time_col: week[1]]
        prices = read_data.ohlc_form(data_week, f"{week[2][1]}min")["close"]
        ma = SIMPLE_METHODS[week[2][0]](prices, week[2][2])
        signals = get_vector_buys(ma, prices)[week[0]: week[1]]

        if signals.empty: continue

        buy_indices = signals[signals == 1].index
        for inicio in buy_indices:
            try:
                pos_inicio = prices.index.get_loc(inicio)
                if pos_inicio >= window_size:
                    v_p = prices.iloc[pos_inicio - window_size : pos_inicio].values
                    v_m = ma.iloc[pos_inicio - window_size : pos_inicio].values
                    
                    # Normalización robusta: distancia porcentual y log-retornos internos
                    secuencia = (v_p - v_m) / v_m
                    
                    ventas = signals[(signals == -1) & (signals.index > inicio)]
                    if not ventas.empty:
                        tramo = prices.loc[inicio : ventas.index[0]]
                        X_sequences.append(secuencia)
                        y_targets.append(tramo.argmin())
            except: continue

    return np.array(X_sequences), np.array(y_targets)

def build_deep_cnn(input_shape: tuple):
    """Arquitectura CNN profunda con Dilated Convolutions y Residual Blocks."""
    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        layers.Conv1D(64, kernel_size=5, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv1D(64, kernel_size=3, padding='same', dilation_rate=2),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling1D(2),
        
        layers.Conv1D(128, kernel_size=3, padding='same', dilation_rate=4),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv1D(128, kernel_size=3, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.GlobalAveragePooling1D(),
        
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    
    opt = optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=opt, loss='huber')
    return model

def train_cnn_velas_model(X: np.ndarray, y: np.ndarray):
    validos = ~np.isnan(X).any(axis=1) & ~np.isnan(y) & (y < 150)
    X, y = X[validos], y[validos]
    
    X = X.reshape((X.shape[0], X.shape[1], 1))
    y_log = np.log1p(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.15, shuffle=True)

    model = build_deep_cnn(input_shape=(X.shape[1], 1))
    
    lr_reducer = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)

    print(f"Entrenando CNN Profunda con {len(X_train)} ejemplos...")
    model.fit(
        X_train, y_train, 
        epochs=300, 
        batch_size=64, 
        validation_split=0.1, 
        callbacks=[lr_reducer, early_stop],
        verbose=1
    )

    preds = np.expm1(model.predict(X_test)).flatten()
    y_real = np.expm1(y_test)
    
    mae = mean_absolute_error(y_real, np.round(preds))
    print(f"\nMAE Alcanzado: {mae:.4f} velas.")

    return model, mae