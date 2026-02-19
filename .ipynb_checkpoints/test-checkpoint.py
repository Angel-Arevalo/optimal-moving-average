from read_data import read_asset, ohlc_form, parse_to_form
from plot_ import main_plot
from use_tecnics import main
from rsi_opt import optimizer
from find_best import best_partition

#data = read_asset("EURUSD_2025-2026.csv")

#best_partition(data, 10)

parse_to_form("EURUSD_2025-2026.csv")
