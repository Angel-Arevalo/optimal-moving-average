import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from use_tecnics import main
from read_data import ohlc_form
from tester import get_rsi, atr_normalized

def create_features_limit_order(ohlc_df: pd.DataFrame) -> pd.DataFrame:
    close = ohlc_df["close"]
    df = pd.DataFrame(index=ohlc_df.index)
    
    df['returns'] = close.pct_change()
    
    df['atr_norm_5'] = atr_normalized(ohlc_df, n=5)

    rsi_3 = get_rsi(close, n=3)
    df['rsi_3'] = rsi_3
    df['rsi_3_accel'] = rsi_3.diff() 
    
    df['curvature'] = close.diff().diff()
    
    # fft_features = get_fourier(close, 20, 3) 
    # df = pd.concat([df, fft_features], axis=1)
    
    return df

def build_limit_order_dataset(real_data: pd.DataFrame, weekly_ma_list: list, wait_candles: int = 5, pip_target: float = 0.0005) -> tuple:

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
            
        ohlc_resampled = ohlc_form(data_slice, f"{candle}min")
        close_series = ohlc_resampled["close"]
        high_series = ohlc_resampled["high"]
        
        features_df = create_features_limit_order(ohlc_resampled)
        
        signals_df = main(method, close_series, params)
        if signals_df.empty:
            continue
            
        sell_signals = signals_df[signals_df['Signals'] == -1].copy()
        
        if sell_signals.empty:
            continue

        future_high_max = high_series.iloc[::-1].rolling(window=wait_candles).max().iloc[::-1].shift(-1)
        sell_signals['future_max_high'] = future_high_max.loc[sell_signals.index]
        
        sell_signals['target'] = (sell_signals['future_max_high'] >= (sell_signals['Prices'] + pip_target)).astype(int)
        
        week_mask = (sell_signals.index >= pd.to_datetime(start_date)) & (sell_signals.index <= slice_end)
        sell_signals_week = sell_signals[week_mask]
        
        common_idx = sell_signals_week.index.intersection(features_df.index)
        if common_idx.empty:
            continue
            
        trades = features_df.loc[common_idx].copy()
        trades['target'] = sell_signals_week.loc[common_idx, 'target']
        
        X_list.append(trades.drop(columns=['target']))
        y_list.append(trades['target'])
        
    if not X_list:
        return pd.DataFrame(), pd.Series()
        
    X_final = pd.concat(X_list).dropna()
    y_final = pd.concat(y_list).loc[X_final.index]
    
    return X_final, y_final

def train_limit_order_model(X: pd.DataFrame, y: pd.Series):

    if X.empty or len(X) < 10:
        print("No hay suficientes datos de señales -1 para entrenar.")
        return None
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
    
    clf = RandomForestClassifier(n_estimators=300, max_depth=10, class_weight='balanced', random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    print("Reporte del modelo Limit Order (Predicción de Rebotes):")
    print(classification_report(y_test, y_pred))
    
    return clf