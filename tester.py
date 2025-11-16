import pandas as pd


# Se retornan los kpi's en una tupla con este orden: hit ratio, risk reward ratio, profit ratio
def backtest_ma(man_back: pd.Series, real_data: pd.Series) -> tuple[float, float, float]:
    vector_buy: pd.Series = get_vector_buys(man_back, real_data)

    prices_and_signals: pd.DataFrame = pd.concat([vector_buy, real_data], axis=1, join="inner")
    prices_and_signals.columns = ["Signals", "Prices"]
    
    prices_and_signals = prices_and_signals[prices_and_signals["Signals"] != 0]
    prices_and_signals["trade"] = (prices_and_signals["Prices"].shift(-1) - prices_and_signals["Prices"])*prices_and_signals["Signals"]
    prices_and_signals["trade"] = prices_and_signals["trade"].fillna(0)

    return (hit_ratio(prices_and_signals["trade"]), 
            rr_ratio(prices_and_signals["trade"]),
            profit_ratio(prices_and_signals["trade"])
            )


def get_vector_buys(man_back: pd.Series, real_data: pd.Series) -> pd.Series:
    pre_man: pd.Series = man_back.shift(1)
    pre_data: pd.Series = real_data.shift(1)
    
    # Señales de cruce de moving average, 
    signal_buy: pd.Series = ((pre_man < pre_data) & (man_back > real_data)).astype(int)
    signal_sell: pd.Series = ((pre_man >= pre_data) & (man_back < real_data)).astype(int)

    # En el vector de compra o venta aparece un 1 como compra, un -1 como venta
    # y 0 indica no hacer nada
    vector_buy: pd.Series = (signal_buy - signal_sell)
    return vector_buy[vector_buy != 0]

# esta es una medida de cuántas veces la ma acierta 
def hit_ratio(trade_resume: pd.Series) -> float:
    return len(trade_resume[trade_resume > 0])/len(trade_resume)

# es una medida que me dice cuánto gano en promedio comparado con las perdidas
def rr_ratio(trade_resume: pd.Series) -> float:
    prom_winner: float = trade_resume[trade_resume > 0].mean()
    prom_losser: float = -trade_resume[trade_resume < 0].mean()
    return prom_winner/prom_losser

# Similar al risk reward pero no en promedio
def profit_ratio(trade_resume: pd.Series) -> float:
    sum_winner: float = trade_resume[trade_resume > 0].sum()
    sum_losser: float = -trade_resume[trade_resume < 0].sum()
    return sum_winner/sum_losser
