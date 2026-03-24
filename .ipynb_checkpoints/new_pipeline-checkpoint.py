import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

HORIZONS = [1, 2, 3, 6, 12]
COST = .4 * .0001
DELTA = 0.0001
THRESHOLDS = np.linspace(0.5, 0.7, 21)

TARGET_VOL = 0.10
MAX_RISK_PER_TRADE = 0.005
STOP_MULT = 1.5
TP_MULT = 2.0
MAX_DRAWDOWN = 0.05

MIN_VOL = 1e-6

def walk_forward_backtest(df):

    results = []

    years = [2020, 2021, 2022, 2023, 2024]

    for year in years:

        train_start = str(year - 4)
        train_end   = str(year - 1)
        test_year   = str(year)

        df_slice = df[train_start:test_year]

        res = run_pipeline(df_slice)

        df_alpha = res["df"]
        horizon = res["horizon"]

        df_regime = apply_regime_filter(df_alpha)

        df_exec, _ = run_execution_layer(df_regime, horizon)

        df_test = df_exec[test_year]

        results.append(df_test)

    df_all = pd.concat(results)

    return df_all

def prepare_features(df):
    price = df['close']

    df['ret_1'] = np.log(price / price.shift(1))
    df['ret_5'] = np.log(price / price.shift(5))
    df['ret_15'] = np.log(price / price.shift(15))
    df['ret_60'] = np.log(price / price.shift(60))

    df['vol_15'] = df['ret_1'].rolling(15).std().shift(1)
    df['vol_60'] = df['ret_1'].rolling(60).std().shift(1)

    ma_50 = price.rolling(50).mean().shift(1)
    std_50 = price.rolling(50).std().shift(1)
    df['z_50'] = (price - ma_50) / std_50

    df['up_move'] = (df['ret_1'] > 0).astype(int)
    df['up_ratio_20'] = df['up_move'].rolling(20).mean().shift(1)

    df['streak'] = (df['ret_1'] > 0).astype(int)
    df['streak'] = df['streak'] * (df['streak'].groupby((df['streak'] != df['streak'].shift()).cumsum()).cumcount() + 1)

    ema_fast = price.ewm(span=10, adjust=False).mean()
    ema_slow = price.ewm(span=50, adjust=False).mean()

    df['ema_diff'] = (ema_fast - ema_slow) / price
    df['ema_slope'] = ema_fast.diff()

    df['hour'] = df.index.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    df['dow'] = df.index.dayofweek
    df['dow_sin'] = np.sin(2 * np.pi * df['dow'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dow'] / 7)

    return df

def create_target(df, H):
    future_ret = np.log(df['close'].shift(-H) / df['close'])
    y = (future_ret > (COST + DELTA)).astype(int)
    return y

def normalize(train, test, forward, features):
    mean = train[features].mean()
    std = train[features].std()

    train[features] = (train[features] - mean) / std
    test[features] = (test[features] - mean) / std
    forward[features] = (forward[features] - mean) / std

    return train, test, forward

def split_data(df):
    train = df['2019':'2023']
    test = df['2024']
    forward = df['2025']
    return train, test, forward

def train_model(X_train, y_train):
    model = LGBMClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def find_best_threshold(model, X, y):
    probs = model.predict_proba(X)[:, 1]

    best_th = 0.5
    best_score = -np.inf

    for th in THRESHOLDS:
        pred = (probs > th).astype(int)
        if pred.sum() < 50:
            continue

        ret = pred * y
        score = ret.mean()

        if score > best_score:
            best_score = score
            best_th = th

    return best_th

def run_pipeline(df):

    df = prepare_features(df)

    best_global = {
        "horizon": None,
        "auc": -np.inf,
        "model": None,
        "features": None,
        "threshold": None
    }

    for H in HORIZONS:
        df['target'] = create_target(df, H)

        df_clean = df.dropna().copy()

        train, test, forward = split_data(df_clean)

        features = [col for col in df.columns if col not in ['target', 'close']]

        train, test, forward = normalize(train, test, forward, features)

        X_train = train[features]
        y_train = train['target']

        X_test = test[features]
        y_test = test['target']

        model = train_model(X_train, y_train)

        probs_test = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probs_test)

        if auc > best_global["auc"]:
            best_global.update({
                "horizon": H,
                "auc": auc,
                "model": model,
                "features": features,
            })


    H = best_global["horizon"]
    df['target'] = create_target(df, H)
    df_clean = df.dropna().copy()

    train, test, forward = split_data(df_clean)
    features = best_global["features"]

    train, test, forward = normalize(train, test, forward, features)

    model = train_model(train[features], train['target'])

    threshold = find_best_threshold(model, test[features], test['target'])

    df_clean['alpha'] = model.predict_proba(df_clean[features])[:, 1]
    df_clean['signal'] = (df_clean['alpha'] > threshold).astype(int)

    return {
        "df": df_clean,
        "model": model,
        "features": features,
        "threshold": threshold,
        "horizon": H,
        "auc": best_global["auc"]
    }

def compute_volatility_regime(df):
    df['vol_30'] = df['ret_1'].rolling(30).std().shift(1)

    vol_q_low = df['vol_30'].rolling(2000).quantile(0.2)
    vol_q_high = df['vol_30'].rolling(2000).quantile(0.8)

    df['vol_regime'] = 0

    df.loc[(df['vol_30'] > vol_q_low) & (df['vol_30'] < vol_q_high), 'vol_regime'] = 1

    return df


def compute_trend_regime(df):
    price = df['close']

    ema_fast = price.ewm(span=20, adjust=False).mean()
    ema_slow = price.ewm(span=100, adjust=False).mean()

    trend_strength = (ema_fast - ema_slow) / price

    # pendiente
    slope = ema_slow.diff()

    df['trend_regime'] = (
        (np.abs(trend_strength) > 0.0005) &  
        (np.abs(slope) > 0)
    ).astype(int)

    return df

def compute_chop_regime(df):
    
    high = df['high']
    low = df['low']
    close = df['close']

    tr = np.maximum(high - low,
         np.maximum(np.abs(high - close.shift(1)),
                    np.abs(low - close.shift(1))))

    atr = tr.rolling(30).mean()

    movement = np.abs(close - close.shift(30))

    chop = atr / movement

    df['chop_regime'] = (chop < 2).astype(int)

    return df

def compute_final_regime(df):
    df['regime'] = (
        (df['vol_regime'] == 1) &
        (df['trend_regime'] == 1) &
        (df['chop_regime'] == 1)
    ).astype(int)

    return df

def apply_regime_filter(df):
    df = compute_volatility_regime(df)
    df = compute_trend_regime(df)
    df = compute_chop_regime(df)
    df = compute_final_regime(df)

    # FILTRO FINAL
    df['filtered_signal'] = df['signal'] * df['regime']

    return df



def compute_position_size(df):
    df['realized_vol'] = df['ret_1'].rolling(60).std().shift(1)

    df['realized_vol_annual'] = df['realized_vol'] * np.sqrt(252 * 24 * 60)

    df['base_size'] = TARGET_VOL / (df['realized_vol_annual'] + MIN_VOL)

    df['base_size'] = df['base_size'].clip(0, 3)

    return df


def apply_risk_constraint(df):
    # stop basado en volatilidad
    df['stop_distance'] = df['realized_vol'] * STOP_MULT

    # tamaño máximo permitido
    df['max_size'] = MAX_RISK_PER_TRADE / (df['stop_distance'] + MIN_VOL)

    df['position_size'] = np.minimum(df['base_size'], df['max_size'])

    return df


def compute_positions(df):
    edge = (df['alpha'] - 0.5) * 2
    df['position'] = df['filtered_signal'] * df['position_size'] * edge

    H = df['horizon'].iloc[0]
    df['future_ret'] = np.log(df['close'].shift(-H) / df['close'])

    # stop-loss / take-profit
    df['stop_loss'] = -df['stop_distance']
    df['take_profit'] = df['stop_distance'] * TP_MULT

    # aplicar límites
    df['trade_ret'] = df['future_ret'].clip(
        lower=df['stop_loss'],
        upper=df['take_profit']
    )

    return df


def compute_equity(df):
    df['strategy_ret'] = df['position'] * df['trade_ret']

    df['equity'] = (1 + df['strategy_ret']).cumprod()

    return df

def apply_drawdown_control(df):
    df['peak'] = df['equity'].cummax()
    df['drawdown'] = (df['equity'] - df['peak']) / df['peak']

    df['active'] = (df['drawdown'] > -MAX_DRAWDOWN).astype(int)

    df['strategy_ret'] = df['strategy_ret'] * df['active']

    df['equity'] = (1 + df['strategy_ret']).cumprod()

    return df

def compute_metrics(df):
    total_return = df['equity'].iloc[-1] - 1

    # CAGR aproximado
    n_years = len(df) / (252 * 24 * 60)
    cagr = (df['equity'].iloc[-1]) ** (1 / n_years) - 1

    max_dd = df['drawdown'].min()

    calmar = cagr / abs(max_dd) if max_dd != 0 else np.nan

    return {
        "total_return": total_return,
        "cagr": cagr,
        "max_dd": max_dd,
        "calmar": calmar
    }


# =========================
# FULL PIPELINE
# =========================

def run_execution_layer(df, horizon):

    df = df.copy()
    df['horizon'] = horizon

    df = compute_position_size(df)
    df = apply_risk_constraint(df)
    df = compute_positions(df)
    df = compute_equity(df)
    df = apply_drawdown_control(df)

    metrics = compute_metrics(df)

    return df, metrics