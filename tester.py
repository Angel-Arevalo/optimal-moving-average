import pandas as pd
from numpy import float64, isnan, sqrt

def backtest(signals_and_prices: pd.DataFrame, calq_sqn: bool = False, shorts: bool = False):
    trade_resume: pd.Series = signals_and_prices["Prices"].diff().fillna(0)

    if shorts:
        trade_resume = -trade_resume[signals_and_prices["Signals"] == 1]
    else:
        trade_resume = trade_resume[signals_and_prices["Signals"] == -1]

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
    signal_buy: pd.Series = ((pre_man <= pre_data) & (man_back > real_data)).astype(int)
    signal_sell: pd.Series = ((pre_man > pre_data) & (man_back <= real_data)).astype(int)

    # En el vector de compra o venta aparece un 1 como compra, un -1 como venta
    # y 0 indica no hacer nada
    vector_buy: pd.Series = (signal_buy - signal_sell)
    vector_buy = vector_buy.fillna(0)

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


def get_total_money(trade_resume: pd.Series) -> float:
    return trade_resume.sum()
