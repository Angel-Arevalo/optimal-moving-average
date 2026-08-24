from datetime import datetime, timezone
import MetaTrader5 as mt5
import pandas as pd
import read_data
import keys
import find_best
import direction_methods
import use_tecnics
import tester_dir

mt5.initialize()

symbol = "EURUSD_"

symbol_info = mt5.symbol_info(symbol)
if symbol_info is None:
    mt5.shutdown()
    raise ValueError(f"No se encontró el símbolo {symbol}")

point = symbol_info.point

utc_from = datetime(2026, 3, 31, 0, 0, tzinfo=timezone.utc)
utc_to = datetime(2026, 8, 21, 23, 59, tzinfo=timezone.utc)

rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M1, utc_from, utc_to)
mt5.shutdown()

df = pd.DataFrame(rates)
df["time"] = pd.to_datetime(df["time"], unit="s")

df = df.rename(columns={"close": "bid"})
df["ask"] = df["bid"] + (df["spread"] * point)

df_result = df[["time", "bid", "ask"]].set_index("time")
print(df_result.loc[datetime(2026, 6, 5, 0, 45): datetime(2026, 6, 5, 1, 5)])
print(df_result.loc[datetime(2026, 8, 7, 16, 30): datetime(2026, 8, 7, 16, 50)])
keys.fill_ohlc_dict(df_result)

_, _, ohlc = read_data.ohlc_form(df_result, 5)

t = use_tecnics.main("SMA", ohlc["close"], 10, True, df_result)

kef = direction_methods.DIR_METHODS["KEF"]({"candle": 63, "window": 10, "follow_tend": .5})

t, p = direction_methods._split_signals_and_change(t, kef, True, df_result)
print(tester_dir.sqn(t, p, True), tester_dir.fsr(t, p, True))
