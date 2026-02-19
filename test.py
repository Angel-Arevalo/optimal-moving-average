from read_data import read_asset, ohlc_form
from plot_ import main_plot
from use_tecnics import main
from rsi_opt import optimizer
data = read_asset("Data_total_EURUSD.csv")

ohlc = ohlc_form(data, "7min")
ma = main("TRIMA", 100, ohlc)
print(optimizer.main(ohlc["close"], ma))
