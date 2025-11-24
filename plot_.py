import  matplotlib.pyplot as plt
from typing import Union
import pandas as pd
from read_data import read_asset, ohlc_form
from use_tecnics import main
import os


def main_plot(asset: Union[str, pd.DataFrame], lookback: int, candle: int, method: str) -> None:
    if isinstance(asset, str):
        data: pd.DataFrame = read_asset(asset)

    if isinstance(asset, pd.DataFrame):
        data: pd.DataFrame = asset

    resample_asset: pd.DataFrame = ohlc_form(data, str(candle) + "min")
    resample_asset: pd.Series = main(method, lookback, resample_asset)

    plt.figure()
    plt.plot(resample_asset, label="Señal")
    plt.plot(data["Precio Spot"], label="Spot")
    plt.legend()
    plt.xlabel("Tiempo")
    plt.ylabel("Precio / Señal")
    plt.tight_layout()

    filename: str = f"{method}_{lookback}_{candle}.png"
    path = os.path.join("Imagenes", filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Imagen guarda en {path}")
