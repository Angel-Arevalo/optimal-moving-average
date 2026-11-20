import numpy as np
import pandas as pd

from direction_methods import trend_at_entries
from find_best_dir import f
from prueba_cadena_markov.direction_comparison import apply_gates


def test_direction_uses_only_detector_bars_already_closed():
    closes = pd.date_range("2024-01-01 04:00", periods=3, freq="4h")
    regime = pd.Series([True, False, True], index=closes)
    entries = pd.to_datetime(["2024-01-01 03:00", "2024-01-01 04:00",
                              "2024-01-01 07:00", "2024-01-01 08:00"])
    assert trend_at_entries(regime, entries).tolist() == [False, True, True, False]
    pd.testing.assert_series_equal(trend_at_entries(regime, entries),
                                   trend_at_entries(regime.iloc[:2], entries))


def test_four_arms_intersect_at_reference_entries():
    times = pd.date_range("2024-01-01", periods=4, freq="h")
    hmm = pd.DataFrame({"entry_time": times, "allow_long_entry": [True, True, False, False]})
    direction = pd.Series([True, False, True, False], index=times)
    gates, _ = apply_gates(hmm, direction[::-1])
    assert gates["media"].all()
    np.testing.assert_array_equal(gates["direccion"], [True, False, True, False])
    np.testing.assert_array_equal(gates["hmm"], [True, True, False, False])
    np.testing.assert_array_equal(gates["direccion_hmm"], [True, False, False, False])


def test_discard_objective_ignores_counterfactual_short_performance():
    base = (.6, 1.1, 1.3, 100, .8, .7)
    filtered = (.65, 1.2, 1.8, 75, .6, .9, 25, 2.)
    assert f(base, (*filtered, 0.), filter_side="revert") == f(base, (*filtered, 100.), filter_side="revert")
    assert f(base, (*filtered, 0.)) != f(base, (*filtered, 100.))


def test_incomplete_aggregate_cannot_fill_without_a_source_close():
    from prueba_cadena_markov.direction_comparison import executable_strategy_view
    from prueba_cadena_markov.market import strategy_view

    times = pd.date_range("2024-01-01", periods=11, freq="h").delete(5)
    price = [10, 10, 10, 10, 5, 15, 16, 14, 13, 12]
    bars = pd.DataFrame({k: price for k in ("open", "high", "low", "close")}, index=times)
    ma = ("SMA", 2, 2)
    _, raw = strategy_view(bars, ma)
    _, executable = executable_strategy_view(bars, ma, "h", pd.Timedelta(hours=1))
    assert pd.Timestamp("2024-01-01 06:00") in raw.index
    assert pd.Timestamp("2024-01-01 06:00") not in executable.index
    pd.testing.assert_frame_equal(executable, raw.loc[executable.index])
