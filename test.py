from find_best import best_partition, calls, log_prices, candles, realized_variance
from read_data import read_asset
from plot_ import main_plot
data = read_asset("Data_total_BTCUSD.csv")

# ejemplo de uso de testeo particionado
calls = 300
print(log_prices(data))
print(realized_variance(data))
