import pandas as pd
import numpy as np
from use_tecnics import main  
from read_data import ohlc_form
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calcula el Relative Strength Index."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    for i in range(period, len(gain)):
        avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (period - 1) + gain.iloc[i]) / period
        avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (period - 1) + loss.iloc[i]) / period
        
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calcula el Average True Range (Requiere OHLC)."""
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=period).mean()
    # Suavizado EMA para ATR
    return atr.ewm(alpha=1/period, adjust=False).mean()

def calculate_williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calcula el Williams %R (Ubicación del rango)."""
    highest_high = df['high'].rolling(window=period).max()
    lowest_low = df['low'].rolling(window=period).min()
    wr = (highest_high - df['close']) / (highest_high - lowest_low) * -100
    return wr


def create_features(ohlc_df: pd.DataFrame) -> pd.DataFrame:

    df = ohlc_df.copy()
    close = df['close']
    
    features = pd.DataFrame(index=df.index)
    
    returns = close.pct_change()
    features['raw_volatility'] = returns.rolling(14).std()
    features['curvature'] = close.diff().diff()

    sma_long = close.rolling(window=200, min_periods=100).mean()
    features['trend_context_ratio'] = close / sma_long

    atr_short = calculate_atr(df, period=14)
    atr_long = calculate_atr(df, period=100)
    features['volatility_regime_ratio'] = atr_short / atr_long

    features['range_location_wr'] = calculate_williams_r(df, period=14)
    
    features['momentum_rsi'] = calculate_rsi(close, period=14)

    return features.ffill().fillna(0)


def build_weekly_dataset(real_data: pd.DataFrame, weekly_ma_list: list, pip_value: float = 0.0001) -> tuple:
    X_list = []
    y_list = []
    
    feature_cols = [
        'raw_volatility', 'curvature', 
        'trend_context_ratio', 'volatility_regime_ratio', 
        'range_location_wr', 'momentum_rsi'
    ]
    
    for start_date, end_date, ma_config in weekly_ma_list:
        method = ma_config[0]
        candle = int(ma_config[1])
        params = ma_config[2:]
        
        max_needed_lookback = max(max(params) if params else 100, 200) 
        pad_days = pd.Timedelta(days=max(15, (candle * max_needed_lookback * 2) / 1440))
        
        slice_start = pd.to_datetime(start_date) - pad_days
        slice_end = pd.to_datetime(end_date) + pd.Timedelta(days=1)
        
        data_slice = real_data[(real_data.index >= slice_start) & (real_data.index <= slice_end)]
        if data_slice.empty or len(data_slice) < max_needed_lookback:
            continue

        ohlc_resampled = ohlc_form(data_slice, f"{candle}min")
        
        if ohlc_resampled.empty or len(ohlc_resampled) < 200: 
            continue

        features_df = create_features(ohlc_resampled)
        
        signals_df = main(method, ohlc_resampled['close'], params)
        if signals_df.empty:
            continue
            
        signals_df['next_close'] = signals_df['Prices'].shift(-1)
        
        week_mask = (signals_df.index >= pd.to_datetime(start_date)) & (signals_df.index <= slice_end)
        signals_week = signals_df[week_mask]
        
        common_idx = signals_week.index.intersection(features_df.index)
        if common_idx.empty:
            continue
            
        trades = features_df.loc[common_idx].copy()
        trades['Signals'] = signals_week.loc[common_idx, 'Signals']
        trades['Prices'] = signals_week.loc[common_idx, 'Prices']
        trades['next_close'] = signals_week.loc[common_idx, 'next_close']
        
        trades = trades.dropna(subset=['next_close'])
        if trades.empty:
            continue
            
        trades['profit_raw'] = (trades['next_close'] - trades['Prices']) * trades['Signals']
        trades['target'] = (trades['profit_raw'] > pip_value).astype(int)
        
        X_list.append(trades[feature_cols])
        y_list.append(trades['target'])
        
    if not X_list:
        return pd.DataFrame(), pd.Series()
        
    X_final = pd.concat(X_list).dropna()
    y_final = pd.concat(y_list).loc[X_final.index]
    
    return X_final, y_final


def train_model(X: pd.DataFrame, y: pd.Series):
    if X.empty:
        print("No hay datos para entrenar.")
        return None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
    
    clf = RandomForestClassifier(n_estimators=500, max_depth=12, class_weight='balanced', random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    print("Reporte de Clasificación (Test Set):")
    print(classification_report(y_test, y_pred))
    
    importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nImportancia de las Features:")
    print(importances)
    
    return clf


def filter_signals_with_ml(real_data: pd.DataFrame, weekly_ma_list: list, modelo, umbral: float = 0.5) -> pd.DataFrame:
    final_results = []
    
    feature_cols = [
        'raw_volatility', 'curvature', 'trend_context_ratio', 
        'volatility_regime_ratio', 'range_location_wr', 'momentum_rsi'
    ]

    for start_date, end_date, ma_config in weekly_ma_list:
        method = ma_config[0]
        candle = int(ma_config[1])
        params = ma_config[2:]
        
        pad_days = pd.Timedelta(days=20)
        slice_start = pd.to_datetime(start_date)
        slice_end = pd.to_datetime(end_date) + pd.Timedelta(days=1)
        
        data_slice = real_data[slice_start - pad_days : slice_end]
        if len(data_slice) < 200:
            continue

        ohlc_resampled = ohlc_form(data_slice, f"{candle}min")
        features_df = create_features(ohlc_resampled)
        
        signals_df = main(method, ohlc_resampled['close'], params)
        if signals_df.empty:
            continue
        
        week_mask = (signals_df.index >= slice_start) & (signals_df.index <= slice_end)
        signals_week = signals_df[week_mask].copy()
        
        if signals_week.empty:
            continue

        if signals_week['Signals'].iloc[0] == -1:
            signals_week = signals_week.iloc[1:]
        
        common_idx = signals_week.index.intersection(features_df.index)
        trades_to_eval = signals_week.loc[common_idx]
        
        entradas = trades_to_eval[trades_to_eval['Signals'] == 1]
        
        if not entradas.empty:
            X_eval = features_df.loc[entradas.index, feature_cols]
            probs = modelo.predict_proba(X_eval)[:, 1]

            aprobados_dict = dict(zip(entradas.index, probs >= umbral))
            
            indices_semana = []
            en_operacion_aprobada = False
            
            for idx, row in signals_week.iterrows():
                sig = row['Signals']
                if sig == 1:
                    if aprobados_dict.get(idx, False):
                        indices_semana.append(idx)
                        en_operacion_aprobada = True
                    else:
                        en_operacion_aprobada = False
                elif sig == -1:
                    if en_operacion_aprobada:
                        indices_semana.append(idx)
                        en_operacion_aprobada = False
            
            if indices_semana:
                final_results.append(signals_week.loc[indices_semana])

    if not final_results:
        return pd.DataFrame(columns=['Signals', 'Prices'])
        
    df_final = pd.concat(final_results)

    df_final = df_final[~df_final.index.duplicated(keep='first')]
    
    return df_final.sort_index()

def get_signals_and_prices(data: pd.DataFrame, ma_week: list) -> pd.DataFrame:
    """Genera y concatena el vector de señales de todas las semanas configuradas."""
    all_weeks = []
    
    for week in ma_week:
        # Obtener datos de la semana + buffer previo para la MA
        time_col = week[0] - pd.Timedelta(weeks=1)
        data_week_raw = data.loc[time_col: week[1]]
        
        # Resample y cálculo de MA
        method_name, candle, params = week[2][0], week[2][1], week[2][2]
        velas = read_data.ohlc_form(data_week_raw, f"{candle}min")["close"]
        
        df_week = main(method_name, velas, params)
        df_week = df_week.loc[week[0]: week[1]]
        
        if len(df_week) > 1:
            # Asegurar que la semana empiece con 1 y termine con -1
            if df_week["Signals"].iloc[0] == -1: df_week = df_week.iloc[1:]
            if df_week["Signals"].iloc[-1] == 1: df_week = df_week.iloc[:-1]
            all_weeks.append(df_week)
            
    if not all_weeks: return pd.DataFrame()
    
    result = pd.concat(all_weeks).sort_index()
    return result[~result.index.duplicated()]