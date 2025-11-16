import pandas as pd


# Se lee la info para tener un DataFrame con la forma time | spot
def read_asset(asset_name: str) -> pd.DataFrame:
    asset_info: pd.DataFrame =  pd.read_csv(f"Data//{asset_name}")
    asset_info["time"] = pd.to_datetime(asset_info["time"])
    return asset_info.set_index("time").dropna()

# Se organiza la info en el formato ohlc, 
#                                   open-high-low-close
#
# Intervalos válidos: 1min, 5min, 15min, 1H, ...
def ohlc_form(asset_name:str, time_rule: str) -> pd.DataFrame:
    return read_asset(asset_name)["Precio Spot"].resample(time_rule).ohlc().ffill().bfill()
