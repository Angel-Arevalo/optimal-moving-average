import matplotlib.pyplot as plt
from typing import Union
import pandas as pd
from read_data import read_asset, ohlc_form
from use_tecnics import main
import os


def main_plot(asset: Union[str, pd.DataFrame], lookback: int, candle: int, method: str) -> None:
    if isinstance(asset, str):
        data: pd.DataFrame = read_asset(asset)
    else:
        data: pd.DataFrame = asset

    ohlc: pd.DataFrame = ohlc_form(data, f"{candle}min")
    signal: pd.Series = main(method, lookback, ohlc)

    fig, ax_price = plt.subplots(figsize=(11, 5))

    ax_price.plot(ohlc.index, ohlc["close"], label="Spot", linewidth=1.4)
    ax_price.set_xlabel("Tiempo")
    ax_price.set_ylabel("Precio")

    ax_sig = ax_price.twinx()
    ax_sig.plot(signal.index, signal, label="Signal", linewidth=1.2, linestyle="--", color="red")
    ax_sig.set_ylabel("Señal")

    h1, l1 = ax_price.get_legend_handles_labels()
    h2, l2 = ax_sig.get_legend_handles_labels()
    ax_price.legend(h1 + h2, l1 + l2, loc="best")

    fig.tight_layout()

    filename: str = f"{method}_{lookback}_{candle}.png"
    path = os.path.join("Imagenes", filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Imagen guardada en {path}")

