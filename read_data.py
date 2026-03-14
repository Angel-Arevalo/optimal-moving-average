import pandas as pd
from typing import Union

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
def ohlc_form(asset: Union[str, pd.DataFrame], time_rule: str) -> pd.DataFrame:
    if isinstance(asset, str):
        return read_asset(asset)["Precio Spot"].resample(time_rule).ohlc().ffill().bfill()

    return asset["Precio Spot"].resample(time_rule).ohlc().ffill()

