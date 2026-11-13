from pathlib import Path
from numbers import Integral
import numpy as np
import pandas as pd


def read_asset(data, asset: str = None) -> pd.DataFrame:
    if isinstance(data, (str, Path)):
        path = Path(data)
        readers = {".csv": pd.read_csv, ".parquet": pd.read_parquet, ".json": pd.read_json, ".feather": pd.read_feather}
        if path.suffix.lower() in (".nc", ".netcdf"):
            import xarray as xr
            with xr.open_dataarray(path) as source:
                return read_asset(source.load(), asset)
        if path.suffix.lower() not in readers:
            raise ValueError("Formato no disponible: use CSV, Parquet, JSON, Feather o NetCDF")
        data = readers[path.suffix.lower()](path)

    if type(data).__module__.startswith("xarray"):
        if "asset" in data.dims:
            assets = data.coords["asset"].values
            if asset is None and len(assets) != 1:
                raise ValueError("Seleccione un activo con asset=...")
            data = data.sel(asset=assets[0] if asset is None else asset)
        if hasattr(data, "data_vars"):
            if "field" in data.dims and len(data.data_vars) == 1:
                data = data[next(iter(data.data_vars))]
            elif "field" in data.dims:
                raise ValueError("Seleccione una variable del Dataset")
        if "field" in data.dims:
            data = data.transpose("time", "field").to_pandas()
        else:
            data = data.to_dataframe() if hasattr(data, "data_vars") else data.to_dataframe(name=data.name or "close")

    if isinstance(data, pd.Series):
        data = data.unstack("field") if isinstance(data.index, pd.MultiIndex) else data.to_frame("close")
    elif not isinstance(data, pd.DataFrame):
        try:
            data = pd.DataFrame(data)
        except (TypeError, ValueError) as exc:
            raise ValueError("Entrada no disponible: proporcione datos tabulares con fechas y OHLC") from exc
    df = data.copy()

    if isinstance(df.index, pd.MultiIndex):
        if "asset" in df.index.names:
            assets = df.index.get_level_values("asset").unique()
            if asset is None and len(assets) != 1:
                raise ValueError("Seleccione un activo con asset=...")
            df = df.xs(assets[0] if asset is None else asset, level="asset")
        if "field" in df.index.names and df.shape[1] == 1:
            df = df.iloc[:, 0].unstack("field")
    if isinstance(df.columns, pd.MultiIndex):
        if asset is not None:
            levels = [i for i in range(df.columns.nlevels) if asset in df.columns.get_level_values(i)]
            if len(levels) != 1:
                raise ValueError("No se encontró el activo en las columnas")
            df = df.xs(asset, axis=1, level=levels[0])
        else:
            levels = [i for i in range(df.columns.nlevels) if df.columns.get_level_values(i).nunique() == 1]
            for level in reversed(levels):
                df.columns = df.columns.droplevel(level)
        if isinstance(df.columns, pd.MultiIndex):
            raise ValueError("Seleccione un activo con asset=...")

    df.columns = [str(col).strip().lower().strip("<>") for col in df.columns]
    df = df.rename(columns={"date": "time", "datetime": "time", "timestamp": "time", "volume": "vol"})
    if "asset" in df.columns:
        assets = df["asset"].dropna().unique()
        if asset is None and len(assets) != 1:
            raise ValueError("Seleccione un activo con asset=...")
        df = df[df["asset"] == (assets[0] if asset is None else asset)].drop(columns="asset")
    if "time" in df.columns:
        df = df.set_index("time")
    if df.empty:
        raise ValueError("No hay datos para el activo seleccionado")
    if not isinstance(df.index, pd.DatetimeIndex) and pd.api.types.is_numeric_dtype(df.index.dtype):
        raise ValueError("La entrada necesita fechas; convierta timestamps numéricos con su unidad explícita")
    df.index = pd.to_datetime(df.index, errors="raise", utc=True).tz_convert(None)
    df.index.name = "time"
    if df.index.hasnans or df.columns.has_duplicates:
        raise ValueError("Fechas inválidas o columnas duplicadas")
    sides = [f"{side}_{field}" for side in ("bid", "ask") for field in ("open", "high", "low", "close")]
    quotes = any(col in df for col in ["bid", "ask", *sides])
    if quotes:
        if "bid" in df and "ask" in df:
            for side in ("bid", "ask"):
                for field in ("open", "high", "low", "close"):
                    if f"{side}_{field}" not in df:
                        df[f"{side}_{field}"] = df[side]
        if not all(col in df for col in sides):
            raise ValueError("Proporcione bid y ask o bid/ask OHLC completos")
        df[sides] = df[sides].apply(pd.to_numeric, errors="raise")
        for field in ("open", "high", "low", "close"):
            df[field] = (df[f"bid_{field}"] + df[f"ask_{field}"]) / 2
    if "close" not in df:
        raise ValueError("No se encontró close ni bid/ask")
    ohlc = ["open", "high", "low", "close"]
    if not any(col in df for col in ohlc[:-1]):
        df = df.assign(open=df["close"], high=df["close"], low=df["close"])
    if not all(col in df for col in ohlc):
        raise ValueError("Proporcione OHLC completo o únicamente close")
    prices = ohlc + (sides if quotes else [])
    cols = prices + [col for col in ("vol", "is_liquid") if col in df]
    df = df[cols].apply(pd.to_numeric, errors="raise").sort_index(kind="stable").dropna(subset=prices)
    if df.empty or not np.isfinite(df[prices].to_numpy()).all() or (df[prices] <= 0).any().any():
        raise ValueError("Los precios deben ser positivos y finitos")
    for prefix in (["", "bid_", "ask_"] if quotes else [""]):
        frame = df[[prefix + field for field in ohlc]].set_axis(ohlc, axis=1)
        if ((frame.high < frame[["open", "close", "low"]].max(axis=1)) | (frame.low > frame[["open", "close", "high"]].min(axis=1))).any():
            raise ValueError("OHLC inconsistente: revise high y low")
    if quotes and any((df[f"ask_{field}"] < df[f"bid_{field}"]).any() for field in ohlc):
        raise ValueError("Ask no puede ser menor que bid")
    if df.index.has_duplicates:
        if quotes:
            df = df.groupby(level=0, sort=True).agg(aggregation(df))
        else:
            raise ValueError("Fechas duplicadas")
    if "is_liquid" in df:
        df["is_liquid"] = df["is_liquid"].fillna(0).gt(0)
    return mid_ohlc(df)


def mid_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    if "ask_close" in df:
        for field in ("open", "high", "low", "close"):
            df[field] = (df[f"bid_{field}"] + df[f"ask_{field}"]) / 2
    return df


def aggregation(df: pd.DataFrame) -> dict:
    rules = {"open": "first", "high": "max", "low": "min", "close": "last", "vol": "sum", "is_liquid": "last"}
    return {col: rules[col.rsplit("_", 1)[-1]] if col != "is_liquid" else "last" for col in df}


def ohlc_form(data, time_rule: int = 1, temporality: str = "auto", asset: str = None, origin="epoch") -> pd.DataFrame:
    return resample_asset(read_asset(data, asset), time_rule, temporality, origin)


def resample_asset(df: pd.DataFrame, time_rule: int = 1, temporality: str = "auto", origin="epoch") -> pd.DataFrame:
    if temporality not in ("auto", "min", "h", "d") or not isinstance(time_rule, Integral) or time_rule < 1:
        raise ValueError("Use auto, min, h o d y una temporalidad positiva")
    intraday = df.index.normalize().has_duplicates
    if temporality == "auto":
        temporality = "min" if intraday else "d"
    if temporality != "d" or intraday:
        delta = pd.Timedelta(time_rule, unit=temporality)
        steps = df.index.to_series().diff().dropna()
        if len(steps) and steps.median() > delta:
            raise ValueError("La temporalidad solicitada es menor que la resolución de entrada")
        result = df.resample(delta, closed="right", label="right", origin=origin).agg(aggregation(df))
        return mid_ohlc(result.loc[result.index <= df.index[-1]].dropna(subset=["close"]))
    if time_rule == 1:
        return df
    size = len(df) // time_rule * time_rule
    if not size:
        return df.iloc[:0]
    result = df.iloc[:size].groupby(np.arange(size) // time_rule).agg(aggregation(df))
    result.index = df.index[time_rule - 1:size:time_rule]
    return mid_ohlc(result)


def execution_prices(signals: pd.Series, data: pd.DataFrame) -> pd.Series:
    if "ask_close" not in data:
        return data["close"].reindex(signals.index, method="ffill")
    frame = data.reindex(signals.index, method="ffill")
    return pd.Series(np.where(signals == 1, frame.ask_close, frame.bid_close), index=signals.index)
