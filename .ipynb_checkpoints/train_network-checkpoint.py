import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from typing import Tuple
from scipy.stats import skew, kurtosis

import read_data
from use_tecnics import main, SIMPLE_METHODS
from tester import get_vector_buys

def extraer_propiedades_matematicas(ventana_precios: pd.Series, ventana_ma: pd.Series) -> dict:
    p_vals = ventana_precios.to_numpy(dtype=float)
    m_vals = ventana_ma.to_numpy(dtype=float)
    n = len(p_vals)

    if n < 5:
        return {"FFT_Score": 0, "Concavidad": 0, "Pendiente": 0, "Sesgo": 0, 
                "Curtosis": 0, "Area_Separacion": 0, "Curvatura": 0, "Persistencia": 0}

    # --- FFT y Polinomios (Tus métricas actuales) ---
    p_centrado = p_vals - np.mean(p_vals)
    fft_res = np.abs(np.fft.fft(p_centrado))[:n // 2]
    fft_res[0] = 0
    energia_total = np.sum(fft_res)
    score_fft = np.max(fft_res) / energia_total if energia_total > 0 else 0

    x = np.arange(n)
    coefs = np.polyfit(x, p_vals, 2)
    concavidad = coefs[0]
    pendiente = coefs[1]

    cambios_absolutos = np.abs(np.diff(p_vals)).sum()
    desplazamiento_neto = np.abs(p_vals[-1] - p_vals[0])
    curvatura = cambios_absolutos / desplazamiento_neto if desplazamiento_neto > 0 else 0

    distancias = p_vals - m_vals
    persistencia = np.std(distancias)

    retornos = np.diff(p_vals) / p_vals[:-1]
    sesgo = float(skew(retornos)) if len(retornos) > 0 else 0.0
    curtosis_val = float(kurtosis(retornos)) if len(retornos) > 0 else 0.0

    area_separacion = np.trapezoid(distancias)

    return {
        "FFT_Score": score_fft,
        "Concavidad": concavidad,
        "Pendiente": pendiente,
        "Sesgo": sesgo,
        "Curtosis": curtosis_val,
        "Area_Separacion": area_separacion,
        "Curvatura": curvatura,
        "Persistencia": persistencia 
    }

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

def get_signals_and_prices(data: pd.DataFrame, ma_week: list) -> pd.Series:
    signals_and_prices = None
    
    for week in ma_week:
        data_week = get_info_in_week(data, week)

        if signals_and_prices is None:
            signals_and_prices = data_week
        else:
            signals_and_prices = pd.concat([signals_and_prices, data_week])

    return signals_and_prices

def get_features(data: pd.DataFrame, ma_week: list) -> pd.DataFrame:
    total_info = pd.DataFrame()

    for week in ma_week:
        time_col = week[0] - pd.Timedelta(weeks=1)
        data_week = data.loc[time_col: week[1]]

        prices = read_data.ohlc_form(data_week, f"{week[2][1]}min")["close"]

        method_name = week[2][0]
        params = week[2][2]
        
        ma = SIMPLE_METHODS[method_name](prices, params)
    
        data_week = get_vector_buys(ma, prices)
        
        data_week = data_week[week[0]: week[1]]

        if data_week.empty:
            continue

        f_i = data_week.index[0]
        l_i = data_week.index[-1]

        if data_week[f_i] == -1:
            f_i = data_week.index[1] if len(data_week) > 1 else f_i
        if data_week[l_i] == 1:
            l_i = data_week.index[-2] if len(data_week) > 1 else l_i

        data_week = data_week[f_i: l_i]
        
        start_index = data_week[data_week == 1].index
        end_index = data_week[data_week == -1].index
        
        for i in range(len(start_index)):
            inicio = start_index[i]
            fin = end_index[i]

            velas_lookback = week[2][2]

            pos_inicio = prices.index.get_loc(inicio)
            
            if pos_inicio >= velas_lookback:
                ventana_p = prices.iloc[pos_inicio - velas_lookback : pos_inicio]
                ventana_m = ma.iloc[pos_inicio - velas_lookback : pos_inicio]
                props_matematicas = extraer_propiedades_matematicas(ventana_p, ventana_m)
            else:
                continue

            tramo_precios = prices.loc[inicio : fin]
            tramo_ma = ma.loc[inicio : fin]

            velas_a_esperar = tramo_precios.argmin()

            precio_entrada = tramo_precios.iloc[0]
            ma_entrada = tramo_ma.iloc[0]
            distancia_relativa = (precio_entrada - ma_entrada) / ma_entrada

            df_resumen = pd.DataFrame({"Fecha_Señal": [inicio],
                                       "Distancia_MA": [distancia_relativa],
                                       "FFT_Score": [props_matematicas["FFT_Score"]],
                                       "Concavidad": [props_matematicas["Concavidad"]],
                                       "Pendiente": [props_matematicas["Pendiente"]],
                                       "Sesgo": [props_matematicas["Sesgo"]],
                                       "Curtosis": [props_matematicas["Curtosis"]],
                                       "Area_Separacion": [props_matematicas["Area_Separacion"]],
                                       "Curvatura": [props_matematicas["Curvatura"]],
                                       "Persistencia": [props_matematicas["Persistencia"]],
                                       "Target_Velas": [velas_a_esperar],
                                       "Tramo_ID": [f"{week[0].strftime('%Y-%m-%d')}_{i}"]
                                    })

            if total_info.empty:
                total_info = df_resumen
            else:
                total_info = pd.concat([total_info, df_resumen])

    return total_info

def train_target_velas_model(df_ml: pd.DataFrame) -> Tuple[MLPRegressor, StandardScaler, list]:
    columnas_x = [
        'Distancia_MA', 'FFT_Score', 'Concavidad', 'Pendiente', 
        'Sesgo', 'Curtosis', 'Area_Separacion', 'Curvatura', 'Persistencia'
    ]

    df_clean = df_ml.dropna(subset=columnas_x + ['Target_Velas'])
    
    X = df_clean[columnas_x]
    y = df_clean['Target_Velas']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    red_neuronal = MLPRegressor(
        hidden_layer_sizes=(64, 32), 
        activation='relu', 
        solver='adam', 
        max_iter=5000,           
        learning_rate_init=0.001,
        early_stopping=True,   
        validation_fraction=0.1, 
        n_iter_no_change=100,
        random_state=42
    )

    print(f"Entrenando red neuronal con {len(X_train)} ejemplos...")
    red_neuronal.fit(X_train_scaled, y_train)

    predicciones = red_neuronal.predict(X_test_scaled)
    predicciones_redondeadas = np.maximum(0, np.round(predicciones)) 
    
    mae = mean_absolute_error(y_test, predicciones_redondeadas)
    
    print("\n--- Resultados del Entrenamiento ---")
    print(f"Error Absoluto Medio (MAE): {mae:.2f} velas de margen de error.")
    
    df_comparacion = pd.DataFrame({
        'Real': y_test.values[:5],
        'Predicción': predicciones_redondeadas[:5]
    })
    print("\nMuestra de las primeras 5 predicciones en Test:")
    print(df_comparacion)

    return red_neuronal, scaler, columnas_x