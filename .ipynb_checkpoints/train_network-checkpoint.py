import pandas as pd
import numpy as np
from use_tecnics import main
from read_data import ohlc_form
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def filter_signals_with_ml(signals_and_prices: pd.DataFrame, real_data, modelo, umbral: float = 0.5) -> pd.DataFrame:
    filtered_df = signals_and_prices.copy()

    signal_mask = filtered_df['Signals'] != 0
    signal_indices = filtered_df[signal_mask].index
    
    if len(signal_indices) == 0:
        return filtered_df 

    start_date = signal_indices.min()
    end_date = signal_indices.max()

    margen_tiempo = pd.Timedelta(minutes=30)
    
    if isinstance(real_data, pd.DataFrame):
        close_series = real_data.iloc[:, 0]
    else:
        close_series = real_data
        
    close_slice = close_series.loc[start_date - margen_tiempo : end_date]
    
    features_df = create_features(close_slice)

    valid_indices = signal_indices.intersection(features_df.index)
    
    if len(valid_indices) == 0:
        filtered_df['Signals'] = 0
        return filtered_df
    
    cols = ['volatility', 'curvature', 'fft_1', 'fft_2', 'fft_3']
    X_to_predict = features_df.loc[valid_indices, cols].fillna(0)
    
    probabilidades = modelo.predict_proba(X_to_predict)[:, 1]
    
    decisiones = (probabilidades >= umbral).astype(int)
    decisiones_dict = dict(zip(valid_indices, decisiones))
    
    filtered_df['Signals'] = 0
    
    en_posicion = False
    ultimo_indice_compra = None
    
    for idx in signal_indices:
        orig_signal = signals_and_prices.loc[idx, 'Signals']
        
        if orig_signal == 1 and not en_posicion:
            decision_ia = decisiones_dict.get(idx, 0)
            
            if decision_ia == 1:
                filtered_df.loc[idx, 'Signals'] = 1
                en_posicion = True
                ultimo_indice_compra = idx
                
        elif orig_signal == -1 and en_posicion:
            filtered_df.loc[idx, 'Signals'] = -1
            en_posicion = False
            ultimo_indice_compra = None
            
    if en_posicion and ultimo_indice_compra is not None:
        filtered_df.loc[ultimo_indice_compra, 'Signals'] = 0
        
    return filtered_df


def get_fourier(series: pd.Series, window_size: int, num_harmonics: int) -> pd.DataFrame:
    fft_df = pd.DataFrame(index=series.index)
    for i in range(1, num_harmonics + 1):
        fft_df[f'fft_{i}'] = np.nan
        
    values = series.values
    for i in range(window_size, len(values)):
        window = values[i-window_size:i]
        fft_vals = np.abs(np.fft.fft(window))
        for h in range(1, num_harmonics + 1):
            fft_df.iloc[i, h-1] = fft_vals[h]
            
    return fft_df

def create_features(series: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame(index=series.index)
    df['close'] = series
    df['returns'] = series.pct_change()
    df['volatility'] = df['returns'].rolling(14).std()
    df['curvature'] = series.diff().diff()
    
    fft_features = get_fourier(series, 20, 3)
    return pd.concat([df, fft_features], axis=1)

def build_weekly_dataset(real_data: pd.DataFrame, weekly_ma_list: list, pip_value: float = 0.0001) -> tuple:
    X_list = []
    y_list = []
    
    for start_date, end_date, ma_config in weekly_ma_list:
        method = ma_config[0]
        candle = int(ma_config[1])
        params = ma_config[2:]
        
        max_lookback = max(params) if params else 100
        pad_days = pd.Timedelta(days=max(10, (candle * max_lookback * 2) / 1440))
        
        slice_start = pd.to_datetime(start_date) - pad_days
        slice_end = pd.to_datetime(end_date) + pd.Timedelta(days=1)
        
        data_slice = real_data[(real_data.index >= slice_start) & (real_data.index <= slice_end)]
        if data_slice.empty:
            continue
            
        ohlc_resampled = ohlc_form(data_slice, f"{candle}min")["close"]
        features_df = create_features(ohlc_resampled)
        
        signals_df = main(method, ohlc_resampled, params)
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
        
        feature_cols = ['volatility', 'curvature', 'fft_1', 'fft_2', 'fft_3']
        X_list.append(trades[feature_cols])
        y_list.append(trades['target'])
        
    if not X_list:
        return pd.DataFrame(), pd.Series()
        
    X_final = pd.concat(X_list).dropna()
    y_final = pd.concat(y_list).loc[X_final.index]
    
    return X_final, y_final

def train_model(X: pd.DataFrame, y: pd.Series):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
    
    clf = RandomForestClassifier(n_estimators=300, max_depth=10, class_weight='balanced', random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    return clf