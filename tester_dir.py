import pandas as pd
from tester import Metrics, trade_returns, summarize, mae as trade_mae


def backtest(rev: pd.DataFrame, tend: pd.DataFrame, data: pd.DataFrame, shorts: bool = False) -> Metrics:
    trades = pd.concat([trade_returns(rev, shorts), trade_returns(tend, not shorts)]).sort_index()
    count_rev, count_tend = len(rev) // 2, len(tend) // 2
    total = count_rev + count_tend
    adverse = trade_mae(rev, data, shorts) * count_rev + trade_mae(tend, data, not shorts) * count_tend
    return summarize(trades, adverse / total if total else 0.0)
