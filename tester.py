import pandas as pd
from numpy import float64, isnan, sqrt

def backtest(signals_and_prices: pd.DataFrame, metrics: bool = False, fee: float = 0.0001):
    prices = signals_and_prices["Prices"].values
    signals = signals_and_prices["Signals"].values

    position = 0
    entry_price = 0
    returns = []

    for i in range(len(signals)):
        sig = signals[i]

        if sig == 1 and position == 0:
            position = 1
            entry_price = prices[i]

        elif sig == -1 and position == 1:
            exit_price = prices[i]

            ret = (exit_price / entry_price - 1) - 2 * fee
            returns.append(ret)

            position = 0

    returns = pd.Series(returns)

    if len(returns) == 0:
        if metrics:
            return -10, 0, 0, 0, returns, 0, 0, 0
        return -10, 0, 0, 0, returns

    mu = returns.mean()
    sigma = returns.std()

    sharpe = -10 if sigma == 0 else mu / sigma
    tr = len(returns)

    if metrics:
        ht = hit_ratio(returns)
        rr = rr_ratio(returns)
        pr = profit_ratio(returns)

        return ht, rr, pr, tr

    return sharpe, mu, sigma, tr, returns


def get_vector_buys(man_back: pd.Series, real_data: pd.Series) -> pd.Series:
    pre_man: pd.Series = man_back.shift(1)
    pre_data: pd.Series = real_data.shift(1)
    
    # Señales de cruce de moving average, 
    signal_buy: pd.Series = ((pre_man <= pre_data) & (man_back > real_data)).astype(int)
    signal_sell: pd.Series = ((pre_man > pre_data) & (man_back <= real_data)).astype(int)

    # En el vector de compra o venta aparece un 1 como compra, un -1 como venta
    # y 0 indica no hacer nada
    vector_buy: pd.Series = (signal_buy - signal_sell)
    vector_buy = vector_buy.fillna(0)

    #vector_buy = vector_buy.shift(1)
    return vector_buy[vector_buy != 0]

def hit_ratio(trade_resume: pd.Series) -> float:
    if len(trade_resume) == 0:
        return 0.0
    
    ganadoras = (trade_resume > 0).sum()
    return ganadoras / len(trade_resume)

def rr_ratio(trade_resume: pd.Series) -> float:
    winners = trade_resume[trade_resume > 0]
    losers = trade_resume[trade_resume < 0]

    if winners.empty or losers.empty:
        return 0.0

    prom_winner = winners.mean()
    prom_losser = -losers.mean()
    return prom_winner / prom_losser

def profit_ratio(trade_resume: pd.Series) -> float:
    winners = trade_resume[trade_resume > 0]
    losers = trade_resume[trade_resume < 0]

    if winners.empty or losers.empty or losers.sum() == 0:
        return 0.0

    return winners.sum() / (-losers.sum())

def get_rsi(data: pd.Series, n: int) -> pd.Series:
    delta = data.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/n, adjust=False).mean()

    rs = avg_gain.div(avg_loss.replace(0, 1e-10))
    rsi = 100 - (100 / (1 + rs))

    return rsi

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    
    high = df["high"]
    low = df["low"]
    close = df["close"]
    
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    return tr.ewm(alpha=1/n, adjust=False).mean()

def atr_normalized(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return  atr(df, n)/df["close"]

def get_total_money(trade_resume: pd.Series) -> float:
    return trade_resume.sum()
