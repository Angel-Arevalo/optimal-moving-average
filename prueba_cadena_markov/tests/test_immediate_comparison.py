import pandas as pd

from prueba_cadena_markov import immediate_comparison as immediate
from prueba_cadena_markov.config import RunConfig


def test_both_optimizers_only_receive_26_weeks_ending_at_test_start(tmp_path, monkeypatch):
    start = pd.Timestamp("2024-01-01")
    first = start - pd.Timedelta(weeks=26)
    index = pd.date_range(first-pd.Timedelta(days=1), start+pd.Timedelta(days=7), freq="h")
    data = pd.DataFrame({k: range(1, len(index)+1) for k in ("open","high","low","close")}, index=index)
    path = tmp_path/"ohlc.parquet"
    data.to_parquet(path)
    seen = []
    def fit(train, *args, **kwargs):
        seen.append(train.index)
        return "EMA", 1, 4
    def fit_direction(train, **kwargs):
        seen.append(train.index)
        assert kwargs["filter_side"] == "revert"
        return {}, {}
    monkeypatch.setattr(immediate, "optimize", fit)
    monkeypatch.setattr(immediate, "opti_dir", fit_direction)
    record, _ = immediate.train_week(start, path=path, config=RunConfig(), trials=100)
    expected = pd.date_range(first,start,freq="h",inclusive="left")
    assert len(seen) == 2 and all(s.equals(expected) for s in seen)
    assert record["entrenamiento_fin"] == record["prueba_inicio"]
    assert record["ultima_vela_entrenamiento"]+pd.Timedelta(hours=1) == start
