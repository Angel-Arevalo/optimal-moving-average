import pandas as pd
import read_data
import find_best
from use_tecnics import main
from tester import test_ma_rsi
from plot_ import main_plot

nombre_activo: str = "EURUSD.parquet"
data: pd.DataFrame = read_data.read_asset(nombre_activo)

find_best.methods = {"EMA"}
find_best.candles = 1
method, lookback, candle, n_rsi = find_best.optimize_ma_rsi(data)

print(method, lookback, candle, n_rsi)

ohlc = read_data.ohlc_form(data, str(candle) + "min")

moving_averague: pd.Series = main(method, ohlc, lookback)

ht, rr, pr, tr = test_ma_rsi(ohlc["close"], moving_averague, n_rsi)

print(ht, rr, pr, tr)
print(-find_best.func_to_opt_rsi(data, method, lookback, candle, n_rsi))
#main_plot(data, lookback, candle, method)
