import pandas as pd
import numpy as np

from tester import mae as mae_t
import keys

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

    if len(trades) < 2:
        return 0.0

    mu: float = np.mean(trades)
    des: float = np.std(trades, ddof=1)

    if des == 0 or np.isnan(des):
        return 0.0

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
        return 0

    return dd_reduction / profit_reduction

def max_drawdown(trade_resume: np.ndarray) -> float:
    equity = np.cumsum(trade_resume)
    peak = np.maximum.accumulate(equity)
    dd = peak - equity

    return np.max(dd) if len(dd) > 0 else 0.0

def profit(trade_resume: np.ndarray) -> float:
    return np.sum(trade_resume)

def hit_ratio(rev: pd.DataFrame, tend: pd.DataFrame, short: bool) -> float:
    rev_resume: np.array = rev["Prices"].diff().fillna(0)
    tend_resume: np.array = tend["Prices"].diff().fillna(0)

    if short:
        rev_resume = -rev_resume[rev["Signals"] == 1].to_numpy()
        tend_resume = tend_resume[tend["Signals"] == -1].to_numpy()
    else:
        rev_resume = rev_resume[rev["Signals"] == -1].to_numpy()
        tend_resume = -tend_resume[tend["Signals"] == 1].to_numpy()

    trades = np.concatenate([rev_resume, tend_resume])
    if len(trades) == 0:
        return 0

    win: pd.Series = trades[trades > 0]

    return len(win)/len(trades)

def risk_reward(rev: pd.DataFrame, tend: pd.DataFrame, short: bool) -> float:
    rev_resume: np.array = rev["Prices"].diff().fillna(0)
    tend_resume: np.array = tend["Prices"].diff().fillna(0)

    if short:
        rev_resume = -rev_resume[rev["Signals"] == 1].to_numpy()
        tend_resume = tend_resume[tend["Signals"] == -1].to_numpy()
    else:
        rev_resume = rev_resume[rev["Signals"] == -1].to_numpy()
        tend_resume = -tend_resume[tend["Signals"] == 1].to_numpy()

    trades = np.concatenate([rev_resume, tend_resume])
    win: pd.Series = trades[trades > 0]
    loss: pd.Series = trades[trades <= 0]

    if len(win) == 0 or len(loss) == 0 or np.sum(loss) == 0:
        return 0.0

    return win.mean() / (-loss.mean())

def profit_factor(rev: pd.DataFrame, tend: pd.DataFrame, short: bool) -> float:
    rev_resume: np.array = rev["Prices"].diff().fillna(0)
    tend_resume: np.array = tend["Prices"].diff().fillna(0)

    if short:
        rev_resume = -rev_resume[rev["Signals"] == 1].to_numpy()
        tend_resume = tend_resume[tend["Signals"] == -1].to_numpy()
    else:
        rev_resume = rev_resume[rev["Signals"] == -1].to_numpy()
        tend_resume = -tend_resume[tend["Signals"] == 1].to_numpy()

    trades = np.concatenate([rev_resume, tend_resume])
    win: pd.Series = trades[trades > 0]
    loss: pd.Series = trades[trades <= 0]

    if len(win) == 0 or len(loss) == 0 or np.sum(loss) == 0:
        return 0.0

    return win.sum() / (-loss.sum())

def mae(rev: pd.DataFrame, tend: pd.DataFrame, short: bool, candle: int) -> float:
    if short:
        rev_ohlc: float = mae_t(rev, keys.bid_cache[candle]["high"], short)
        tend_ohlc: float = mae_t(tend, keys.ask_cache[candle]["low"], not short)
    else:
        rev_ohlc: float = mae_t(rev, keys.ask_cache[candle]["low"], short)
        tend_ohlc: float = mae_t(tend, keys.bid_cache[candle]["high"], not short)

    if len(rev) == 0 and len(tend) == 0:
        return 0

    return (rev_ohlc + tend_ohlc)/(len(rev)//2 + len(tend)//2)
