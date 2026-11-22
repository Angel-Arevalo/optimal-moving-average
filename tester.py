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


def get_vector_buys(man_back: pd.Series, real_data: pd.Series, nooh_data: pd.DataFrame, shorts: bool = False, daily: bool = False, *, weekly_closes: bool = True, side: str = None) -> pd.Series:

    if side is None:
        from keys import side
    if side not in ("tend", "revert"):
        raise ValueError("side must be 'tend' or 'revert'")

    pre_man: pd.Series = man_back.shift(1)
    pre_data: pd.Series = real_data.shift(1)

    signal_buy: pd.Series = ((pre_man <= pre_data) & (man_back > real_data)).astype(int)
    signal_sell: pd.Series = ((pre_man > pre_data) & (man_back <= real_data)).astype(int)

    vector_buy: pd.Series = (signal_buy - signal_sell).fillna(0)#.shift(1).dropna()

    vector_buy = vector_buy[(vector_buy != 0)]

    if side == "tend":
        vector_buy = -vector_buy

    entry_sig = -1 if shorts else 1
    definitive_vector: list[pd.Series] = []

    if nooh_data is None:
        raise ValueError("No se puede hacer sin la data original")

    if not weekly_closes:
        # Para el fitness sólo cuentan operaciones completas. No inventar una
        # salida al terminar la muestra ni al cambiar de semana.
        if not vector_buy.empty and vector_buy.iloc[0] != entry_sig:
            vector_buy = vector_buy.iloc[1:]
        if not vector_buy.empty and vector_buy.iloc[-1] == entry_sig:
            vector_buy = vector_buy.iloc[:-1]
        return vector_buy

    for fecha_domingo, señales_semana in vector_buy.groupby(pd.Grouper(freq='W')):

        if señales_semana.empty:
            continue

        if señales_semana.iloc[0] == -entry_sig:
            señales_semana = señales_semana.iloc[1:]


        if señales_semana.empty:
            continue

        if señales_semana.iloc[-1] == entry_sig:
            fecha_lunes = fecha_domingo.normalize() - pd.Timedelta(days=6)
            fin_semana = fecha_lunes + pd.DateOffset(days=7)
            velas_semana = real_data.loc[(real_data.index >= fecha_lunes) & (real_data.index < fin_semana)]
            ultimo_disponible = velas_semana.index[-1]
            if ultimo_disponible == señales_semana.index[-1]:
                señales_semana = señales_semana.iloc[:-1]
            else:
                señales_semana = señales_semana.copy()
                señales_semana.loc[ultimo_disponible] = -entry_sig

        if not señales_semana.empty:
            definitive_vector.append(señales_semana)

    if not definitive_vector:
        return vector_buy.iloc[:0]

    return pd.concat(definitive_vector)

def hit_ratio(trade_resume: pd.Series) -> float:
    if len(trade_resume) == 0:
        return 0.0

    ganadoras = (trade_resume > 0).sum()
    return ganadoras / len(trade_resume)

def rr_ratio(trade_resume: pd.Series) -> float:
    winners = trade_resume[trade_resume > 0]
    losers = trade_resume[trade_resume <= 0]

    if winners.empty or losers.empty:
        return 0.0

    prom_winner = winners.mean()
    prom_losser = -losers.mean()
    return prom_winner / prom_losser

def profit_ratio(trade_resume: pd.Series) -> float:
    winners = trade_resume[trade_resume > 0]
    losers = trade_resume[trade_resume <= 0]

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

    for i in range(len(indices)//2):
        precio_entrada: float = signals_and_prices.loc[indices[2*i], "Prices"]
        periodo: pd.Series = ohlc_data.loc[indices[2*i]: indices[2*i +1]]

        peak_val: float = periodo.max() if short else periodo.min()
 
        trade_mae = (-1 if short else 1) * (precio_entrada - peak_val)
        trade_mae = (trade_mae * 100)/precio_entrada

        mae_val += trade_mae

    return mae_val/(len(signals_and_prices)//2)

