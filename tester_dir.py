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

def msr(rev: pd.DataFrame, tend: pd.DataFrame) -> float:
    return len(tend)/len(rev)

def DEF(rev: pd.DataFrame, tend: pd.DataFrame, org_df: pd.DataFrame, short: bool) -> float:
    rev_resume = rev["Prices"].diff().fillna(0)
    tend_resume = tend["Prices"].diff().fillna(0)
    org_tr = org_df["Prices"].diff().fillna(0)

    if short:
        org_tr_s = -org_tr[org_df["Signals"] == 1]
        rev_resume_s = -rev_resume[rev["Signals"] == 1]
        tend_resume_s = tend_resume[tend["Signals"] == -1]
    else:
        org_tr_s = org_tr[org_df["Signals"] == -1]
        rev_resume_s = rev_resume[rev["Signals"] == -1]
        tend_resume_s = -tend_resume[tend["Signals"] == 1]

    new_trades_series = pd.concat([rev_resume_s, tend_resume_s]).sort_index()
    new_trades = new_trades_series.to_numpy()
    org_tr_arr = org_tr_s.to_numpy()

    new_dd = max_drawdown(new_trades)
    prev_dd = max_drawdown(org_tr_arr)

    new_profit = new_trades.sum()
    prev_profit = org_tr_arr.sum()

    dd_reduction = (prev_dd - new_dd) / prev_dd if prev_dd != 0 else 0.0
    profit_reduction = (prev_profit - new_profit) / prev_profit if prev_profit != 0 else 0.0

    if profit_reduction == 0:
        return float("inf") if dd_reduction > 0 else 0.0

    return dd_reduction / profit_reduction

def max_drawdown(trade_resume: np.ndarray) -> float:
    equity = np.cumsum(trade_resume)
    peak = np.maximum.accumulate(equity)
    dd = peak - equity

    return np.max(dd) if len(dd) > 0 else 0.0

def profit(trade_resume: np.ndarray) -> float:
    return np.sum(trade_resume)
