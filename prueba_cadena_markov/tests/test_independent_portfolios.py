import pandas as pd

from prueba_cadena_markov.independent_portfolios import replay_week


def test_flat_account_can_accept_new_ma_while_another_holds_old_ma():
    times = pd.date_range("2024-01-08 01:00", periods=3, freq="h")
    current = pd.DataFrame({"Signals": [1, -1], "Prices": [12., 15.]}, index=times[:2])
    carried = pd.DataFrame({"Signals": [-1], "Prices": [18.]}, index=times[2:])
    old_position = dict(entry_time=pd.Timestamp("2024-01-07"), entry_price=10., window=0)
    holding, flat = [], []
    assert replay_week(old_position, holding, current, carried, 1, times[0], ("EMA", 1, 5)) is None
    assert replay_week(None, flat, current, carried.iloc[:0], 1, times[0], ("EMA", 1, 5)) is None
    assert [(r["entry_price"], r["exit_price"]) for r in holding] == [(10., 18.)]
    assert [(r["entry_price"], r["exit_price"]) for r in flat] == [(12., 15.)]
