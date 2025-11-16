from read_data import ohlc_form
from use_tecnics import main
import pandas as pd


resample_data: pd.DataFrame = ohlc_form("Data_total_US500.csv", "2min")
print(main("SMA", 3, resample_data))
