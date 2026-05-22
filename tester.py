import pandas as pd
from numpy import float64, isnan, sqrt, nan

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


def get_vector_buys(man_back: pd.Series, real_data: pd.Series, nooh_data: pd.DataFrame = None, shorts: bool = False) -> pd.Series:
    pre_man: pd.Series = man_back.shift(1)
    pre_data: pd.Series = real_data.shift(1)

    # Señales de cruce de moving average, 
    signal_buy: pd.Series = ((pre_man <= pre_data) & (man_back > real_data)).astype(int)
    signal_sell: pd.Series = ((pre_man > pre_data) & (man_back <= real_data)).astype(int)

    # En el vector de compra o venta aparece un 1 como compra, un -1 como venta
    # y 0 indica no hacer nada
    vector_buy: pd.Series = (signal_buy - signal_sell)
    vector_buy = vector_buy[(vector_buy != 0) & (vector_buy.notna())]

    if nooh_data is not None:
        friday: pd.Series = vector_buy[vector_buy.index.dayofweek == 4]
        friday = friday.groupby(friday.index.date).last()

        signals: list = []
        time_sig: list = []
        for date, sig in friday.items():
            if shorts:
                if sig == -1:
                    signals.append(1)

                    time_sig.append(pd.Timestamp(f"{date} 23:50:00"))
                else:
                    if sig == 1:
                        signals.append(-1)

                        time_sig.append(pd.Timestamp(f"{date} 23:50:00"))

        print(time_sig, signals)

        if len(signals) > 0:
            friday = pd.Series(signals, index=time_sig)

            vector_buy = pd.concat([vector_buy, friday]).sort_index()
            vector_buy = vector_buy[~vector_buy.index.duplicated(keep='last')]

            vector_buy = vector_buy[vector_buy != vector_buy.shift(1)]

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
