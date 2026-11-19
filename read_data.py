import pandas as pd
from typing import Union

# Se lee la info para tener un DataFrame con la forma time | spot
def read_asset(asset_name: str) -> pd.DataFrame:
    if asset_name[-1] == "v":
        asset_info: pd.DataFrame = pd.read_csv(asset_name)
    elif asset_name[-1] == "t":
        asset_info: pd.DataFrame = pd.read_parquet(asset_name)

    asset_info.columns = ["time", "Precio Spot"]
    asset_info["time"] = pd.to_datetime(asset_info["time"])
    return asset_info.set_index("time").dropna()


# Se organiza la info en el formato ohlc,
#                                   open-high-low-close
#

OHLC = ["open", "high", "low", "close"]


OHLC_AGG = {"open": "first", "high": "max", "low": "min", "close": "last"}


def ohlc_resolution(index: pd.DatetimeIndex) -> pd.Timedelta:
    if not index.normalize().has_duplicates:
        return pd.Timedelta(days=1)
    return index.to_series().diff().min()


def prepare_asset(asset) -> pd.DataFrame:
    if isinstance(asset, str):
        df = pd.read_csv(asset) if asset.endswith(".csv") else pd.read_parquet(asset)
    elif isinstance(asset, pd.DataFrame):
        df = asset
    elif hasattr(asset, "dims") and hasattr(asset, "to_pandas"):
        # Quantiacs entrega un DataArray con dimensiones time, field y asset.
        if "asset" in asset.dims:
            if asset.sizes["asset"] != 1:
                raise ValueError("Seleccione un activo con data.sel(asset=simbolo)")
            asset = asset.isel(asset=0)
        df = asset.transpose("time", "field").to_pandas()
    else:
        raise TypeError("Se requiere un DataFrame, una ruta o un DataArray de Quantiacs")

    df = df.rename(columns=lambda col: str(col).strip("<>").lower())
    for col in ["time", "date"]:
        if col in df.columns:
            df = df.set_index(col)
            break

    if not isinstance(df.index, pd.DatetimeIndex):
        if pd.api.types.is_numeric_dtype(df.index.dtype):
            raise ValueError("Se requiere un índice temporal o una columna time/date")
        df.index = pd.to_datetime(df.index)

    df = df.sort_index()
    if {"bid", "ask"}.issubset(df.columns):
        return df[["bid", "ask"]]

    if not set(OHLC).issubset(df.columns):
        raise ValueError("Se requieren bid y ask, o las cuatro columnas open, high, low y close")

    df = df[OHLC].dropna()
    if df.empty or df.index.hasnans or df.index.has_duplicates:
        raise ValueError("Se requiere una vela OHLC por timestamp de un único activo")
    return df


def ohlc_form(asset, time_rule: int, temporality: str = "min"):
    df = prepare_asset(asset)
    return prepared_ohlc(df, time_rule, temporality)


def prepared_ohlc(df: pd.DataFrame, time_rule: int, temporality: str):
    if temporality not in ["min", "h", "d"] or time_rule < 1:
        raise ValueError("Use una duración positiva en min, h o d")

    quotes = "bid" in df
    if temporality == "d":
        # Una unidad es una sesión observada, también con datos intradía.
        if quotes:
            df = df[df.index.dayofweek < 5]
        sessions = df.index.normalize()
        days = sessions.unique()
        groups = sessions.factorize(sort=False)[0] // time_rule
        completed = len(days) // time_rule
        closes = days[time_rule - 1::time_rule]
        if quotes:
            bid = df["bid"].groupby(groups).ohlc().iloc[:completed]
            ask = df["ask"].groupby(groups).ohlc().iloc[:completed]
            bid.index = ask.index = closes
            return bid, ask, (bid + ask) / 2
        bars = df.groupby(groups).agg(OHLC_AGG).iloc[:completed]
        bars.index = closes
        return bars.copy(), bars.copy(), bars

    if not quotes:
        duration = pd.Timedelta(time_rule, unit=temporality)
        source = ohlc_resolution(df.index)
        if duration < source or duration % source:
            raise ValueError("La duración solicitada debe ser múltiplo de la vela OHLC de origen")
        offset = (df.index[0] - df.index[0].normalize()) % source
        bins = df.resample(str(time_rule) + temporality, origin="start_day", offset=offset)
        bars = bins.agg(OHLC_AGG).dropna()
        # El último intervalo sólo puede usarse cuando termina la última vela fuente.
        bars = bars[bars.index + duration <= df.index[-1] + source]
        return bars.copy(), bars.copy(), bars

    ohlc_bid_total = []
    ohlc_ask_total = []

    resample_rule = str(time_rule) + temporality

    for fecha_domingo, data_sem in df.groupby(pd.Grouper(freq='W')):
        fecha_viernes = (fecha_domingo - pd.Timedelta(days=2)).replace(hour=23, minute=59)
        fecha_lunes = (fecha_domingo - pd.Timedelta(days=6)).replace(hour=00, minute=00)

        ohlc_bid_sem = (
            df.loc[fecha_lunes: fecha_viernes]["bid"]
            .resample(resample_rule)
            .ohlc()
        )
        ohlc_ask_sem = (
            df.loc[fecha_lunes: fecha_viernes]["ask"]
            .resample(resample_rule)
            .ohlc()
        )

        ohlc_bid_total.append(ohlc_bid_sem)
        ohlc_ask_total.append(ohlc_ask_sem)

    bid_ohlc: pd.DataFrame = pd.concat(ohlc_bid_total).dropna()
    ask_ohlc: pd.DataFrame = pd.concat(ohlc_ask_total).dropna()
    if temporality == "min":
        bid_ohlc, ask_ohlc = filtrar_horario_activo(bid_ohlc, ask_ohlc, df.index, time_rule)

    mid_ohlc = pd.DataFrame({"open": (bid_ohlc["open"] + ask_ohlc["open"]) / 2,
                              "high": (bid_ohlc["high"] + ask_ohlc["high"]) / 2,
                              "low": (bid_ohlc["low"] + ask_ohlc["low"]) / 2,
                              "close": (bid_ohlc["close"] + ask_ohlc["close"]) / 2})

    return bid_ohlc, ask_ohlc, mid_ohlc


def filtrar_horario_activo(bid_ohlc: pd.DataFrame, ask_ohlc: pd.DataFrame, asset_index: pd.DatetimeIndex, candle_minutes: int = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    horarios = pd.Series(asset_index.time, index=asset_index)
    rango_por_dia = horarios.groupby(asset_index.normalize()).agg(['min', 'max'])

    dias_cierre = bid_ohlc.index.normalize()
    horas_cierre = bid_ohlc.index.time

    hora_min = dias_cierre.map(rango_por_dia['min'])
    hora_max = dias_cierre.map(rango_por_dia['max'])

    mask = (~hora_min.isna()) & (hora_min <= horas_cierre) & (horas_cierre <= hora_max)

    return bid_ohlc[mask], ask_ohlc[mask]
