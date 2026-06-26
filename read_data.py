import pandas as pd
from typing import Union
import polars as pl

# Se lee la info para tener un DataFrame con la forma time | spot
def read_asset(asset_name: str) -> pd.DataFrame:
    if asset_name[-1] == "v":
        asset_info: pd.DataFrame =  pd.read_csv(asset_name)
    elif asset_name[-1] == "t":
        asset_info: pd.DataFrame = pd.read_parquet(asset_name)

    asset_info.columns = ["time", "Precio Spot"]
    asset_info["time"] = pd.to_datetime(asset_info["time"])
    return asset_info.set_index("time").dropna()

# Se organiza la info en el formato ohlc, 
#                                   open-high-low-close
#
def ohlc_form(asset: Union[str, pd.DataFrame], time_rule: int, is_bid: bool = False, include_low: bool = False) -> pd.DataFrame:

    if not is_bid:
        if isinstance(asset, str):
            return read_asset(asset)["Precio Spot"].resample(str(time_rule)+"min").ohlc().ffill().bfill()
        return asset["Precio Spot"].resample(str(time_rule) + "min").ohlc().ffill()

    if isinstance(asset, str):
        if asset.endswith(".csv"):
            df = pd.read_csv(asset)
        else:
            df = pd.read_parquet(asset)
    else:
        df = asset.reset_index()

    actual_cols = df.columns
    rename_map = {}
    if "bid" in actual_cols:
        rename_map["bid"] = "bid"
    elif "<BID>" in actual_cols:
        rename_map["<BID>"] = "bid"
    if "ask" in actual_cols:
        rename_map["ask"] = "ask"
    elif "<ASK>" in actual_cols:
        rename_map["<ASK>"] = "ask"

    if "bid" not in rename_map.values() or "ask" not in rename_map.values():
        raise ValueError(f"No se encontraron columnas de Bid/Ask. Columnas detectadas: {list(actual_cols)}")

    df = df.rename(columns=rename_map)

    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")

    bid_ohlc = df["bid"].resample(str(time_rule) + "min").ohlc().ffill().bfill()
    ask_ohlc = df["ask"].resample(str(time_rule) + "min").ohlc().ffill().bfill()

    df_close = pd.DataFrame({
        "bid": bid_ohlc["close"],
        "ask": ask_ohlc["close"],
    })

    df_close["Precio Spot"] = (df_close["bid"] + df_close["ask"]) / 2

    if not include_low:
        return df_close

    df_low = pd.DataFrame({
        "bid": bid_ohlc["low"],
        "ask": ask_ohlc["low"],
    })

    df_low["Precio Spot"] = (df_low["bid"] + df_low["ask"]) / 2

    df_high = pd.DataFrame({
        "bid": bid_ohlc["high"],
        "ask": ask_ohlc["high"],
    })

    df_high["Precio Spot"] = (df_high["bid"] + df_high["ask"]) / 2

    return df_close, df_low, df_high
