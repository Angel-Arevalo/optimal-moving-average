import read_data
import pandas as pd
from find_best import opti_main
import keys
from tester import backtest
from use_tecnics import main
from tester import mae

keys.calls = 15

data = pd.read_csv("EURUSD.csv", index_col="time")
data.index = pd.to_datetime(data.index)

opti_main(data, True)
