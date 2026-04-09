import read_data
import new_pipeline
import pandas as pd
from find_best import opti_main
import keys
keys.calls = 20
keys.candles = 20
keys.methods = {"SMA", "EMA", "DEMA", "TRIMA"}
opti_main("EURUSD_ask_bid.parquet", True)

