from read_data import ohlc_form
from use_tecnics import main
import pandas as pd
from tester import backtest_ma

ohlc: pd.DataFrame = ohlc_form("Data_total_US500.csv", "5min")
ma: pd.Series = main("MIDPOINT", 5, ohlc)

print(backtest_ma(ma, ohlc["close"]))
