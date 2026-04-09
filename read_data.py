import pandas as pd
from typing import Union
import polars as pl

# Se lee la info para tener un DataFrame con la forma time | spot
def read_asset(asset_name: str) -> pd.DataFrame:
    if asset_name[-1] == "v":
        asset_info: pd.DataFrame =  pd.read_csv(f"Data//{asset_name}")
    elif asset_name[-1] == "t":
        asset_info: pd.DataFrame = pd.read_parquet(f"Data/{asset_name}")
    
    asset_info.columns = ["time", "Precio Spot"]
    asset_info["time"] = pd.to_datetime(asset_info["time"])
    return asset_info.set_index("time").dropna()

# Se organiza la info en el formato ohlc, 
#                                   open-high-low-close
#
# Intervalos válidos: 1min, 5min, 15min, 1H, ...
def ohlc_form(asset: Union[str, pd.DataFrame], time_rule: int, is_bid: bool = False) -> pd.DataFrame:
    if not is_bid:

        if isinstance(asset, str):
            return read_asset(asset)["Precio Spot"].resample(str(time_rule)+"min").ohlc().ffill().bfill()

        return asset["Precio Spot"].resample(time_rule).ohlc().ffill() 

    if isinstance(asset, str):
        if asset.endswith(".csv"):
            lf = pl.scan_csv(f"Data/{asset}")
        else:
            lf = pl.scan_parquet(f"Data/{asset}")
    else:
        lf = pl.from_pandas(asset.reset_index()).lazy()

    lf = lf.with_columns(
        pl.col("time").cast(pl.Datetime)
    ).sort("time")

    df_resampled = (
        lf.group_by_dynamic("time", every=str(time_rule)+"m")
        .agg([
            pl.col("bid").last(),
            pl.col("ask").last()
        ])
        .with_columns(
            ((pl.col("bid") + pl.col("ask")) / 2).alias("Precio Spot")
        )
        .collect()
    )

    return df_resampled.to_pandas().set_index("time").ffill().bfill()

