from find_best import best_partition, calls
from read_data import read_asset
from plot_ import main_plot
data = read_asset("Data_total_BTCUSD.csv")

# ejemplo de uso de testeo particionado
calls = 300
best_partition(data, 5)


# ejmplo de dibujo de precio / moving averague
main_plot(data, 70, 67, "T3")
main_plot(data, 87, 1, "KAMA")
main_plot(data, 101, 5, "SMA")
main_plot(data, 87, 2, "SMA")
