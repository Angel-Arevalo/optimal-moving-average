import pandas as pd

# las series de precios, una moving averague y el factor de rsi para ir calculando
def test_ma_rsi(data: pd.Series, moving_averague: pd.Series, rsi_factor: int) -> tuple[float, float, float, int]:
    signal_cross: pd.Series = get_vector_buys(moving_averague, data)
    rsi: pd.Series = get_rsi(data, rsi_factor)
    
    buy_signal: pd.Series = ((signal_cross == 1) & (rsi > 30)).astype(int)
    sell_signal: pd.Series = ((signal_cross == -1) & (rsi < 70)).astype(int)

    signal_cross = buy_signal - sell_signal

    prices_signals: pd.DataFrame = pd.concat([data, signal_cross], axis=1, join="inner")
    prices_signals.columns = ["Prices", "Signals"]
    
    prices_signals["trade"] = (prices_signals["Prices"].shift(-1) - prices_signals["Prices"])*prices_signals["Signals"]
    prices_signals["trade"] = prices_signals["trade"].fillna(0)
    trade_resume: pd.Series = prices_signals.loc[prices_signals["Signals"] != 0, "trade"]

    return (hit_ratio(trade_resume),
            rr_ratio(trade_resume),
            profit_ratio(trade_resume),
            len(trade_resume)
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
    return vector_buy


def get_rsi(data: pd.Series, n: int) -> pd.Series:
    delta = data.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/n, adjust=False).mean()

    rs = avg_gain.div(avg_loss.replace(0, 1e-10))
    rsi = 100 - (100 / (1 + rs))

    return rsi

# esta es una medida de cuántas veces la ma acierta 
def hit_ratio(trade_resume: pd.Series) -> float:
    if len(trade_resume) == 0:
        return 0
    counter: int = 0

    for trade in trade_resume:
        if trade > 0:
            counter += 1

    return counter/len(trade_resume)

def rr_ratio(trade_resume: pd.Series) -> float:
    winners = trade_resume[trade_resume > 0]
    losers = trade_resume[trade_resume < 0]
    if len(winners) == 0 or len(losers) == 0:
        return 0.0
    return winners.mean() / (-losers.mean())

def profit_ratio(trade_resume: pd.Series) -> float:
    winners = trade_resume[trade_resume > 0]
    losers = trade_resume[trade_resume < 0]
    if winners.sum() == 0 or (-losers.sum()) == 0:
        return 0.0
    return winners.sum() / (-losers.sum())
