import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

HORIZONS = [1, 2, 3, 6, 12]
COST = 0.4 * 0.0001

TARGET_VOL = 0.2
MIN_VOL = 1e-6
ALPHA_SCALE = 15
MAX_LEVERAGE = 1.5


def prepare_features(df):
    df = df.copy()
    price = df['close']

    df['ret_1'] = np.log(price / price.shift(1))
    df['ret_5'] = np.log(price / price.shift(5))
    df['ret_15'] = np.log(price / price.shift(15))
    df['ret_60'] = np.log(price / price.shift(60))

    df['vol_15'] = df['ret_1'].rolling(15).std().shift(1)
    df['vol_60'] = df['ret_1'].rolling(60).std().shift(1)

    ma = price.rolling(50).mean().shift(1)
    std = price.rolling(50).std().shift(1)
    df['z_50'] = (price - ma) / (std + MIN_VOL)

    df['up'] = (df['ret_1'] > 0).astype(int)
    df['up_ratio'] = df['up'].rolling(20).mean().shift(1)

    sign = np.sign(df['ret_1'])
    df['streak'] = sign.groupby((sign != sign.shift()).cumsum()).cumcount() + 1
    df['streak'] *= sign

    ema_fast = price.ewm(span=10).mean()
    ema_slow = price.ewm(span=50).mean()

    df['ema_diff'] = (ema_fast - ema_slow) / price
    df['ema_slope'] = ema_fast.diff()

    df['hour'] = df.index.hour
    df['hour_sin'] = np.sin(2*np.pi*df['hour']/24)
    df['hour_cos'] = np.cos(2*np.pi*df['hour']/24)

    df['dow'] = df.index.dayofweek
    df['dow_sin'] = np.sin(2*np.pi*df['dow']/7)
    df['dow_cos'] = np.cos(2*np.pi*df['dow']/7)

    return df


def create_target(df, H):
    return np.log(df['close'].shift(-H) / df['close'])


def normalize(train, test, features):
    mean = train[features].mean()
    std = train[features].std().replace(0, 1)

    for d in [train, test]:
        d[features] = (d[features] - mean) / std
        d[features] = d[features].replace([np.inf, -np.inf], 0).fillna(0)

    return train, test


def train_model(X, y):
    model = LGBMRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    model.fit(X, y)
    return model


def run_pipeline(df):

    df = prepare_features(df)

    best = {
        "score": -np.inf,
        "horizon": None
    }

    for H in HORIZONS:

        df['target'] = create_target(df, H)
        df_clean = df.dropna()

        train = df_clean.loc['2019':'2023']
        test = df_clean.loc['2024']

        features = [c for c in df.columns if c not in ['target', 'close']]

        train, test = normalize(train, test, features)

        model = train_model(train[features], train['target'])

        pred = model.predict(test[features])
        score = np.mean(pred)

        if score > best["score"]:
            best["score"] = score
            best["horizon"] = H

    if best["horizon"] is None:
        raise ValueError("No se pudo seleccionar horizonte")

    # FINAL MODEL
    H = best["horizon"]

    df['target'] = create_target(df, H)
    df_clean = df.dropna()

    train = df_clean.loc['2019':'2023']
    test = df_clean.loc['2024']
    forward = df_clean.loc['2025']

    features = [c for c in df.columns if c not in ['target', 'close']]

    train, test = normalize(train, test, features)

    model = train_model(train[features], train['target'])

    df_clean['alpha'] = np.nan
    df_clean.loc[test.index, 'alpha'] = model.predict(test[features])
    df_clean.loc[forward.index, 'alpha'] = model.predict(forward[features])

    return df_clean, H


def apply_regime_filter(df):

    df = df.copy()

    df['vol'] = df['ret_1'].rolling(30).std().shift(1)

    q_low = df['vol'].rolling(2000).quantile(0.2)
    q_high = df['vol'].rolling(2000).quantile(0.8)

    df['vol_regime'] = ((df['vol'] > q_low) & (df['vol'] < q_high)).astype(int)

    ema_fast = df['close'].ewm(span=20).mean()
    ema_slow = df['close'].ewm(span=100).mean()

    df['trend_regime'] = ((ema_fast - ema_slow).abs() > 0.0005).astype(int)

    move = np.abs(df['close'] - df['close'].shift(30))
    chop = df['vol'] / (move + MIN_VOL)

    df['chop_regime'] = (chop < 2).astype(int)

    df['regime_score'] = (
        0.4*df['vol_regime'] +
        0.3*df['trend_regime'] +
        0.3*df['chop_regime']
    )

    return df


def run_execution_layer(df, horizon):

    df = df.copy()

    df['realized_vol'] = df['ret_1'].rolling(60).std().shift(1)
    df['realized_vol_annual'] = df['realized_vol'] * np.sqrt(252*24*60)

    df['alpha_scaled'] = np.tanh(df['alpha'] * ALPHA_SCALE)

    df['position'] = df['alpha_scaled'] / (df['realized_vol_annual'] + MIN_VOL)
    df['position'] *= TARGET_VOL
    df['position'] *= (np.abs(df['alpha']) > COST)
    df['position'] *= (0.7 + 0.3 * df['regime_score'])
    df['position'] = df['position'].clip(-MAX_LEVERAGE, MAX_LEVERAGE)

    df['position'] *= df['regime_score']

    df['future_ret'] = np.log(df['close'].shift(-horizon) / df['close'])

    df['strategy_ret'] = df['position'] * df['future_ret'] - np.abs(df['position']) * COST
    df['strategy_ret'] = df['strategy_ret'].fillna(0)

    df['equity'] = (1 + df['strategy_ret']).cumprod()

    peak = df['equity'].cummax()
    dd = (df['equity'] - peak) / peak
    df['position'] *= (1 + 0.4 * dd).clip(0.6, 1)

    df['strategy_ret'] = df['position'] * df['future_ret'] - np.abs(df['position']) * COST
    df['equity'] = (1 + df['strategy_ret']).cumprod()

    return df, compute_metrics(df)


def compute_metrics(df):

    equity = df['equity'].dropna()

    total_return = equity.iloc[-1] - 1
    n_years = len(equity) / (252*24*60)
    cagr = equity.iloc[-1] ** (1/n_years) - 1

    peak = equity.cummax()
    dd = (equity - peak) / peak
    max_dd = dd.min()

    calmar = cagr / abs(max_dd) if max_dd != 0 else np.nan

    return {
        "total_return": total_return,
        "cagr": cagr,
        "max_dd": max_dd,
        "calmar": calmar
    }
