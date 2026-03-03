import find_best
import keys
import read_data
import use_tecnics
import tester
"""
data = read_data.read_asset("EURUSD.parquet")
data = read_data.ohlc_form(data, "97min")

signals = use_tecnics.main("T3", data, 108)

hr, rr, pf, tr = tester.backtest(signals)

print(hr, rr, pf, tr)
"""

find_best.opti_main("Data_total_EURUSD.csv")
