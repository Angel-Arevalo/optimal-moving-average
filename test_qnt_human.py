import pandas as pd
import qnt.data as qndata

from direction_methods import dir_main
from find_best_dir import opti_dir
from read_data import prepare_asset
from use_tecnics import main
import keys 

keys.temp = "d"
keys.side = "tend"
keys.candles = 3

data = qndata.stocks.load_ndx_data(
    assets=['NAS:AAPL'],
    min_date="2011-01-01",
    max_date="2020-12-31",
)

data_test = qndata.stocks.load_ndx_data(
    assets=['NAS:AAPL'],
    min_date="2021-01-01",
    max_date="2022-12-31",
)


data: pd.DataFrame = prepare_asset(data)[["open", "high", "low", "close", "volume"]]
print(data.head())
tripla = opti_dir(data, True, False, 200, filter_side="revert", weekly_closes=False)
print(tripla)

data_test = prepare_asset(data_test)[["open", "high", "low", "close", "volume"]]

data_test = data_test.loc[data_test.index > data.index.max()]
if data_test.empty:
    raise ValueError("No hay datos de prueba posteriores al período de entrenamiento")
data_total = pd.concat([data, data_test])
data_total = data_total.loc[~data_total.index.duplicated(keep="last")].sort_index()

params, metrics = tripla
shorts = False
filter_side = metrics["filter"]
base_side = "tend" if filter_side == "tend" else "revert"
daily = keys.fill_ohlc_dict(data_total)
ma_candle = params["ma_candle"]

signals_total = main(
    params["ma_method"], keys.mid_cache[ma_candle]["close"],
    params["ma_lookback"], ma_candle, data_total, shorts,
    daily=daily, weekly_closes=False, side=base_side,
)

start, end = data_test.index.min(), data_test.index.max()
signals_test = signals_total.loc[start:end]

entry_signal = -1 if shorts else 1
if not signals_test.empty and signals_test["Signals"].iloc[0] != entry_signal:
    signals_test = signals_test.iloc[1:]
if not signals_test.empty and signals_test["Signals"].iloc[-1] == entry_signal:
    signals_test = signals_test.iloc[:-1]

hr, rr, pr, trades, mae, sqn, _, _, _ = dir_main(
    signals_test, data_total, ma_candle, params, shorts, filter_side, daily=daily,
)

print(f"\nKPIs de data_test ({start:%Y-%m-%d} a {end:%Y-%m-%d})")
print(f"Hit Ratio: {hr:.2%}")
print(f"Risk Reward: {rr:.4f}")
print(f"Profit Factor: {pr:.4f}")
print(f"Trades: {trades}")
print(f"MAE: {mae:.5f}%")
print(f"SQN: {sqn:.4f}")
