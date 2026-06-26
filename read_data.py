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
        df = pd.read_csv(asset) if asset.endswith(".csv") else pd.read_parquet(asset)
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time")
    else:
        df = asset.copy()

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

    ohlc_bid_total = []
    ohlc_ask_total = []

    for fecha_domingo, data_sem in df.groupby(pd.Grouper(freq='W')):
        fecha_viernes = (fecha_domingo - pd.Timedelta(days=2)).replace(hour=23, minute=59)
        fecha_lunes   = (fecha_domingo - pd.Timedelta(days=6)).replace(hour=00, minute=00)

        ohlc_bid_sem = df.loc[fecha_lunes: fecha_viernes]["bid"].resample(str(time_rule) + "min").ohlc().ffill().bfill()
        ohlc_ask_sem = df.loc[fecha_lunes: fecha_viernes]["ask"].resample(str(time_rule) + "min").ohlc().ffill().bfill()

        ohlc_bid_total.append(ohlc_bid_sem)
        ohlc_ask_total.append(ohlc_ask_sem)

    bid_total: pd.DataFrame = pd.concat(ohlc_bid_total)
    ask_total: pd.DataFrame = pd.concat(ohlc_ask_total)

    df_resample = pd.DataFrame({"bid": bid_total["close"],
                                "ask": ask_total["close"]})

    df_resample["Precio Spot"] = (df_resample["bid"] + df_resample["ask"])/2

    if not include_low:
        return df_resample

    df_low = pd.DataFrame({"bid": bid_total["low"],
                           "ask": ask_total["low"]})

    df_high = pd.DataFrame({"bid": bid_total["high"],
                            "ask": ask_total["high"]})

    return df_resample, df_low, df_high 

