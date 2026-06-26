import pandas as pd
from numpy import float64, isnan, sqrt, nan

def backtest(signals_and_prices: pd.DataFrame, ohlc_data: pd.DataFrame, calq_sqn: bool = False, shorts: bool = False):
    trade_resume: pd.Series = signals_and_prices["Prices"].diff().fillna(0)

    mae_val: float = mae(signals_and_prices, ohlc_data, shorts)

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

        return hr, rr, pr, tr, mae_val, sqn

    return hr, rr, pr, tr, mae_val


def get_vector_buys(man_back: pd.Series, real_data: pd.Series, nooh_data: pd.DataFrame = None, shorts: bool = False) -> pd.DataFrame:

    pre_man: pd.Series = man_back.shift(1)
    pre_data: pd.Series = real_data.shift(1)

    signal_buy: pd.Series = ((pre_man <= pre_data) & (man_back > real_data)).astype(int)
    signal_sell: pd.Series = ((pre_man > pre_data) & (man_back <= real_data)).astype(int)

    vector_buy: pd.Series = (signal_buy - signal_sell).fillna(0)
    vector_buy = vector_buy[(vector_buy != 0)]

    entry_sig = -1 if shorts else 1
    definitive_vector: pd.Series = pd.Series()

    if nooh_data is not None:

        for fecha_domingo, señales_semana in vector_buy.groupby(pd.Grouper(freq='W')):

            if señales_semana.empty:
                continue

            if señales_semana.iloc[0] == -entry_sig:
                señales_semana = señales_semana.iloc[1:]

            if señales_semana.empty:
                continue

            definitive_vector = pd.concat([definitive_vector, señales_semana])

            if señales_semana.iloc[-1] == entry_sig:
                fecha_viernes = fecha_domingo - pd.Timedelta(days=2)

                velas_viernes = real_data[real_data.index.normalize() == fecha_viernes.normalize()]

                if not velas_viernes.empty:
                    ultimo_real = velas_viernes.index[-1]
                    if ultimo_real not in definitive_vector.index:
                        definitive_vector[ultimo_real] = -entry_sig

    return definitive_vector

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

def mae(signals_and_prices: pd.DataFrame, ohlc_data: pd.DataFrame, short: bool):
    if len(signals_and_prices)//2 == 0:
        return 0

    indices: pd.Series = signals_and_prices.index
    mae_val: float = 0

    # Actualemente se recoore la lista así para tomar
    # todos los pares de inicio-fin del trade
    for i in range(len(indices)//2):
        precio_entrada: float = signals_and_prices.loc[indices[2*i], "Prices"]
        periodo: pd.Series = ohlc_data.loc[indices[2*i]: indices[2*i +1]]

        peak_val: float = periodo["bid"].max() if short else periodo["ask"].min()
 
        # siempre debe calcularse el precio de entrada menos el valor al
        # que se necesita, pero hay que tener en cuenta el signo
        trade_mae = (-1 if short else 1) * (precio_entrada - peak_val)
        trade_mae = (trade_mae * 100)/precio_entrada

        mae_val += trade_mae

    return mae_val/(len(signals_and_prices)//2)
