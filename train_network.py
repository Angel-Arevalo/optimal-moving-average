import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from typing import Tuple

import tester 
import use_tecnics 

# Se pasa el dataframe con las columnas ["Signals", "Prices"]
def train_little_model(ma_result: pd.DataFrame) -> Tuple[MLPClassifier, StandardScaler, list]:
    df_ml = pd.DataFrame(index=ma_result.index)

    velas_futuro = 5
    df_ml['Target'] = (ma_result['Prices'].shift(-velas_futuro) > ma_result['Prices']).astype(int)

    df_ml['Retornos'] = ma_result['Prices'].pct_change()
    df_ml['Volatilidad'] = df_ml['Retornos'].rolling(window=14).std()
    df_ml['RSI'] = tester.get_rsi(ma_result["Prices"], n=14)

    df_ml['Senal_Optimizador'] = ma_result['Signals']

    df_ml['Hora'] = ma_result.index.hour
    df_ml['Dia_Semana'] = ma_result.index.dayofweek

    df_ml = df_ml.dropna()

    columnas_x = ['Retornos', 'Volatilidad', 'RSI', 'Senal_Optimizador', 'Hora', 'Dia_Semana']
    X = df_ml[columnas_x]
    y = df_ml['Target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    red_neuronal = MLPClassifier(hidden_layer_sizes=(64, 32), 
                                activation='relu', 
                                solver='adam', 
                                max_iter=500, 
                                random_state=42)

    red_neuronal.fit(X_train_scaled, y_train)

    return red_neuronal, scaler, columnas_x
