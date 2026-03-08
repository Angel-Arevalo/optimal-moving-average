import pandas as pd
from numpy import float64, isnan, sqrt

# Se retornan los kpi's en una tupla con este orden: hit ratio, risk reward ratio, profit ratio

def backtest(signals_and_prices: pd.DataFrame, calq_sqn: bool = False):
    cost: float = 0.00015
    signals_and_prices["Trade"] = (signals_and_prices["Prices"].shift(-1) - signals_and_prices["Prices"])*signals_and_prices["Signals"]
    signals_and_prices["Trade"] = signals_and_prices["Trade"].fillna(0)

    trade_resume: pd.Series = signals_and_prices.loc[signals_and_prices["Signals"] != 0, "Trade"] - cost

    hr = hit_ratio(trade_resume)
    rr = rr_ratio(trade_resume)
    pr = profit_ratio(trade_resume)
    tr = len(trade_resume)
   
    if calq_sqn:
        mu = trade_resume.mean()
        sigma = trade_resume.std()

        if isnan(sigma) or sigma == 0:
            sqn = -10
        else:
            sqn = sqrt(min(tr, 100)) * (mu / sigma)

        return hr, rr, pr, tr, sqn

    return hr, rr, pr, tr


def get_vector_buys(man_back: pd.Series, real_data: pd.Series) -> pd.Series:
    pre_man: pd.Series = man_back.shift(1)
    pre_data: pd.Series = real_data.shift(1)
    
    # Señales de cruce de moving average, 
    signal_buy: pd.Series = ((pre_man < pre_data) & (man_back > real_data)).astype(int)
    signal_sell: pd.Series = ((pre_man >= pre_data) & (man_back < real_data)).astype(int)

    # En el vector de compra o venta aparece un 1 como compra, un -1 como venta
    # y 0 indica no hacer nada
    vector_buy: pd.Series = (signal_buy - signal_sell)
    vector_buy = vector_buy.shift(1).fillna(0)

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
