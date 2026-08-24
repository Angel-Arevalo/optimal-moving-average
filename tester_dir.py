import pandas as pd
import numpy as np

def fsr(rev: pd.DataFrame, tend: pd.DataFrame, short: bool) -> float:
    trade_resume: pd.Series = tend["Prices"].diff().fillna(0)

    if short:
        trade_resume = -trade_resume[tend["Signals"] == 1]
    else:
        trade_resume = trade_resume[tend["Signals"] == -1]

    return (trade_resume > 0).sum()/(trade_resume <= 0).sum()

def sqn(rev: pd.DataFrame, tend: pd.DataFrame, short: bool) -> float:
    rev_resume: np.array = rev["Prices"].diff().fillna(0)
    tend_resume: np.array = tend["Prices"].diff().fillna(0)

    if short:
        rev_resume = -rev_resume[rev["Signals"] == 1].to_numpy()
        tend_resume = tend_resume[tend["Signals"] == -1].to_numpy()
    else:
        rev_resume = rev_resume[rev["Signals"] == -1].to_numpy()
        tend_resume = -tend_resume[tend["Signals"] == 1].to_numpy()

    trades = np.concatenate([rev_resume, tend_resume])

    mu: float = np.mean(trades)
    des: float = np.std(trades, ddof=1)

    if des == 0:
        return 0

    return np.sqrt(min(len(trades), 100)) * mu/des
