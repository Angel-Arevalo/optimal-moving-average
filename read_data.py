import pandas as pd
from typing import Union

# Se lee la info para tener un DataFrame con la forma time | spot
def read_asset(asset_name: str) -> pd.DataFrame:
    asset_info: pd.DataFrame =  pd.read_csv(f"Data//{asset_name}")
    asset_info["time"] = pd.to_datetime(asset_info["time"])
    return asset_info.set_index("time").dropna()

# Se organiza la info en el formato ohlc, 
#                                   open-high-low-close
#
# Intervalos válidos: 1min, 5min, 15min, 1H, ...
def ohlc_form(asset: Union[str, pd.DataFrame], time_rule: str) -> pd.DataFrame:
    if isinstance(asset, str):
        return read_asset(asset_name)["Precio Spot"].resample(time_rule).ohlc().ffill().bfill()

    return asset["Precio Spot"].resample(time_rule).ohlc().ffill()

# Se asume que el archivo va a tener la forma
# <Date> <Time> <Bid> <Ask> <Last> <Volume> <Flags>
# para conseguir un data frame de la forma
# time spot
import pandas as pd

def parse_to_form(asset: str):

   df = pd.read_csv(
    "Data/" + asset,
    sep=r"\s+"
    )


    # limpiar nombres de columnas
    df.columns = df.columns.str.replace("<", "").str.replace(">", "")
    print(df["<ASK>"])
"""
    # unir fecha y hora
    df["time"] = pd.to_datetime(df["DATE"] + " " + df["TIME"])
 
    # usar mid price (más correcto que solo BID)
    df["Precio Spot"] = (df["BID"] + df["ASK"]) / 2

    df_final = df[["time", "Precio Spot"]].copy()

    df_final.sort_values("time", inplace=True)
    df_final.drop_duplicates("time", inplace=True)

    df_final.to_csv("Data/spot.csv", index=False)
"""
