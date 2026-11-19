import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from prueba_cadena_markov.config import MarkovConfig
from prueba_cadena_markov.model import MarkovRegimeFilter
from prueba_cadena_markov.execution import ContinuousExecution, completed_crossing_pairs
from prueba_cadena_markov.economics import equity_curve, trades_from_signals, contained_trades, comparison_metrics
from prueba_cadena_markov.market import strategy_view, SIMPLE_METHODS


def model():
    result = MarkovRegimeFilter(MarkovConfig(n_components=2))
    result.valid = True
    result.favorable_states = {0}
    result.reason = "ready"
    return result


def signals(times=(), signs=(), prices=()):
    return pd.DataFrame({"Signals": signs, "Prices": prices}, index=pd.DatetimeIndex(times))


def advance(execution, current, carried, f, start, end, window, parameters=("SMA", 1, 4), allow=True):
    p = pd.DataFrame([[1., 0.] if allow else [0., 1.]]*len(current), index=current.index)
    execution.advance(current, carried, p, f, start=pd.Timestamp(start), end=pd.Timestamp(end),
                      window=window, parameters=parameters, history_start=pd.Timestamp("2023-01-01"))


def test_raw_signals_never_invent_weekend_or_terminal_exit(monkeypatch):
    index = pd.date_range("2024-01-01", "2024-01-14 23:00", freq="h")
    close = np.where((index >= "2024-01-05 12:00") & (index < "2024-01-10 12:00"), 90., 110.)
    data = pd.DataFrame({"open": close, "high": close+1, "low": close-1, "close": close}, index=index)
    monkeypatch.setitem(SIMPLE_METHODS, "SMA", lambda p, n: pd.Series(100., index=p.index))
    _, full = strategy_view(data, ("SMA", 1, 4))
    assert full.Signals.tolist() == [1, -1]
    assert full.index.tolist() == [pd.Timestamp("2024-01-05 13:00"), pd.Timestamp("2024-01-10 13:00")]
    _, prefix = strategy_view(data.loc[:"2024-01-07 23:00"], ("SMA", 1, 4))
    assert_frame_equal(prefix, full.iloc[:1], check_freq=False)
    assert completed_crossing_pairs(prefix).empty
    _, legacy = strategy_view(data.loc[:"2024-01-07 23:00"], ("SMA", 1, 4), weekly_closes=True)
    assert legacy.Signals.tolist() == [1, -1]
    assert legacy.index[-1] == pd.Timestamp("2024-01-08")


def test_position_survives_week_model_change_and_invalid_hmm():
    execution = ContinuousExecution(n_components=2)
    f = model()
    first = signals(["2024-01-05 13:00"], [1], [90.])
    advance(execution, first, signals(), f, "2024-01-01", "2024-01-08", 0)
    assert execution.trades("Original").empty
    assert len(execution.open_positions("HMM")) == 1
    # New MA would exit earlier, but the entry MA must remain in charge.
    new = signals(["2024-01-08 02:00", "2024-01-11 10:00"], [-1, 1], [95., 105.])
    old = signals(["2024-01-10 13:00"], [-1], [110.])
    f.valid, f.reason = False, "not_converged"
    advance(execution, new, old, f, "2024-01-08", "2024-01-15", 1, parameters=("EMA", 4, 20))
    a, b = execution.trades("Original"), execution.trades("HMM")
    assert_frame_equal(a, b)
    assert a.iloc[0].exit_time == pd.Timestamp("2024-01-10 13:00")
    assert a.iloc[0].window == 0 and a.iloc[0].exit_window == 1
    assert a.iloc[0].parameters == ("SMA", 1, 4)
    assert a.iloc[0].exit_reason == "ma_cross"
    assert execution.open_positions("HMM").empty
    assert len(execution.open_positions("Original")) == 1
    assert execution.decision_frame("HMM").reason.tolist() == ["accepted", "not_converged"]


def test_rejected_reference_position_cannot_generate_extra_hmm_entries():
    execution = ContinuousExecution(n_components=2)
    f = model()
    first = signals(["2024-01-05 13:00"], [1], [90.])
    advance(execution, first, signals(), f, "2024-01-01", "2024-01-08", 0, allow=False)
    new = signals(["2024-01-09 03:00", "2024-01-11 10:00", "2024-01-12 10:00"], [1, 1, -1], [92., 100., 102.])
    old = signals(["2024-01-10 13:00"], [-1], [110.])
    advance(execution, new, old, f, "2024-01-08", "2024-01-15", 1)
    assert len(execution.decision_frame("HMM")) == 2
    assert len(execution.trades("Original")) == 2
    assert len(execution.trades("HMM")) == 1
    assert execution.trades("HMM").iloc[0].entry_time == pd.Timestamp("2024-01-11 10:00")


def test_terminal_position_marked_not_closed_and_entry_fee_once():
    execution = ContinuousExecution(n_components=2, fee_bps=10)
    raw = signals(["2024-01-07 20:00"], [1], [100.])
    advance(execution, raw, signals(), model(), "2024-01-01", "2024-01-08", 0)
    price = pd.Series([100., 100., 90.], index=pd.DatetimeIndex(["2024-01-01", "2024-01-07 20:00", "2024-01-08"]))
    trades, opened = execution.trades("HMM"), execution.open_positions("HMM")
    curve = equity_curve(price, trades, initial_capital=1000, fee_bps=10, open_positions=opened)
    assert trades.empty and curve.units.iloc[-1] == 1
    assert np.isclose(curve.equity.iloc[-1], 989.9)
    metrics = comparison_metrics("HMM", curve, trades, execution.decision_frame("HMM"), 1000)
    assert metrics["Accepted Entry Signals"] == 1
    assert metrics["Number of Trades"] == 0 and metrics["Open Positions"] == 1
    assert np.isclose(metrics["Open Position PnL"], -10.1)


def test_natural_boundary_crossing_labels_are_purged_not_liquidated():
    raw = signals(["2024-01-05", "2024-01-10", "2024-01-11"], [1, -1, 1], [90., 110., 95.])
    pairs = completed_crossing_pairs(raw)
    assert len(pairs) == 2
    trades = trades_from_signals(pairs)
    assert contained_trades(trades, pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-08")).empty
    assert contained_trades(trades, pd.Timestamp("2024-01-08"), pd.Timestamp("2024-01-15")).empty
    assert len(contained_trades(trades, pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-15"))) == 1


def test_no_same_timestamp_reentry_on_new_ma_after_old_ma_exit():
    execution = ContinuousExecution(n_components=2)
    advance(execution, signals(["2024-01-05"], [1], [90.]), signals(), model(), "2024-01-01", "2024-01-08", 0)
    new = signals(["2024-01-10"], [1], [110.])
    old = signals(["2024-01-10"], [-1], [110.])
    advance(execution, new, old, model(), "2024-01-08", "2024-01-15", 1)
    assert execution.position is None
    assert len(execution.decision_frame("Original")) == 1
