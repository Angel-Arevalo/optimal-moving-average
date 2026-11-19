from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from pandas.testing import assert_frame_equal
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM

import keys
from tester import backtest
from read_data import prepared_ohlc
from use_tecnics import main
from prueba_cadena_markov.config import FEATURES, FeatureConfig, MarkovConfig, RunConfig
from prueba_cadena_markov.market import (build_features, strategy_view, observations,
                                        optimizer_settings, validate_market)
from prueba_cadena_markov.model import MarkovRegimeFilter
from prueba_cadena_markov.economics import (trades_from_signals, gate_signals, calibrate,
                                           contained_trades, equity_curve, comparison_metrics)


def market(n=24*21, start="2023-01-02"):
    rng = np.random.default_rng(33)
    p = 100 + 5*np.sin(np.arange(n)/6) + rng.normal(0, .15, n)
    return pd.DataFrame({"open": p, "high": p+1, "low": p-1, "close": p},
                        index=pd.date_range(start, periods=n, freq="h"))


@pytest.fixture
def manual_filter():
    f = MarkovRegimeFilter(replace(MarkovConfig(), n_components=2))
    f.model = GaussianHMM(n_components=2, covariance_type="diag")
    f.model.n_features = 5
    f.model.startprob_ = np.array([.6, .4])
    f.model.transmat_ = np.array([[.95, .05], [.1, .9]])
    f.model.means_ = np.array([[-1]*5, [1]*5], dtype=float)
    f.model.covars_ = np.ones((2, 5))
    f.scaler = StandardScaler().fit(np.array([[-1]*5, [1]*5]))
    f.valid = True
    f.favorable_states = {0}
    f.reason = "ready"
    return f


def feature_frame(n=100):
    return pd.DataFrame(np.random.default_rng(2).normal(size=(n, 5)), columns=FEATURES,
                        index=pd.date_range("2024-01-01", periods=n, freq="h"))


def test_forward_normalization_prefix_and_future_mutation(manual_filter):
    x = feature_frame()
    p = manual_filter.predict_filtered_proba(x)
    assert_allclose(p.sum(axis=1), 1, atol=1e-12)
    assert_frame_equal(p.iloc[:40], manual_filter.predict_filtered_proba(x.iloc[:40]))
    x.iloc[40:] = 99999
    assert_frame_equal(p.iloc[:40], manual_filter.predict_filtered_proba(x).iloc[:40])


def test_forward_matches_direct_bayes(manual_filter):
    x = feature_frame(5)
    probabilities = manual_filter.predict_filtered_proba(x)
    prior = manual_filter.model.startprob_
    for i, row in enumerate(x.to_numpy()):
        if i:
            prior = prior @ manual_filter.model.transmat_
        density = np.exp(-.5*((row-manual_filter.model.means_)**2).sum(axis=1))
        prior = prior*density
        prior /= prior.sum()
        assert_allclose(prior, probabilities.iloc[i], rtol=1e-12)


def test_online_batch_and_missing_observation(manual_filter):
    x = feature_frame()
    x.iloc[[0, 20, 21, 70]] = np.nan
    batch = manual_filter.predict_filtered_proba(x)
    online = manual_filter.stream()
    streamed = pd.concat([online.update(x.iloc[[i]]) for i in range(len(x))])
    assert_allclose(batch, streamed, atol=1e-12, equal_nan=True)
    assert batch.iloc[20].isna().all()
    with pytest.raises(ValueError, match="advance"):
        online.update(x.iloc[[-1]])


def test_no_smoothing_api(monkeypatch, manual_filter):
    def forbidden(*args, **kwargs):
        raise AssertionError("Noncausal library inference used")
    monkeypatch.setattr(GaussianHMM, "predict", forbidden)
    monkeypatch.setattr(GaussianHMM, "predict_proba", forbidden)
    manual_filter.predict_filtered_proba(feature_frame())


def test_features_formula_constant_and_invalid():
    idx = pd.date_range("2024-01-01", periods=50, freq="h")
    price = pd.Series(np.exp(np.arange(50)*.01), index=idx)
    ma = price*.99
    cfg = FeatureConfig(3, 4, 4)
    x = build_features(price, ma, cfg)
    assert_allclose(x.distance_to_ma.dropna(), -np.log(.99))
    assert_allclose(x.distance_change.dropna(), 0, atol=1e-15)
    assert_allclose(x.ma_slope.dropna(), .01)
    assert_allclose(x.realized_volatility.dropna(), .02)
    assert_allclose(x.efficiency_ratio.dropna(), 1)
    constant = build_features(price*0+1, ma*0+1, cfg)
    assert_allclose(constant.iloc[4:], 0)
    assert constant.iloc[0].isna().any()
    ma.iloc[25] = -1
    assert build_features(price, ma, cfg).iloc[25].isna().all()
    price.iloc[30] = np.inf
    assert build_features(price, ma, cfg).iloc[30].isna().all()


@pytest.mark.parametrize("candle", [1, 2, 3, 4, 6])
def test_multitimeframe_features_prefix_invariant(candle):
    data = market()
    ma, _ = strategy_view(data, ("SMA", candle, 3))
    _, x = observations(data, ma, FeatureConfig(3, 4, 4))
    prefix = data.iloc[:133]
    partial_ma, _ = strategy_view(prefix, ("SMA", candle, 3))
    _, partial = observations(prefix, partial_ma, FeatureConfig(3, 4, 4))
    assert_frame_equal(x.loc[partial.index], partial)
    # Changing an unfinished future bar cannot alter earlier observations.
    changed = data.copy()
    changed.iloc[133:] *= 2
    changed_ma, _ = strategy_view(changed, ("SMA", candle, 3))
    _, changed_x = observations(changed, changed_ma, FeatureConfig(3, 4, 4))
    assert_frame_equal(x.loc[partial.index], changed_x.loc[partial.index])


def test_hourly_ma_unavailable_before_close():
    data = market(n=500)
    data.index = pd.date_range("2024-01-01", periods=500, freq="min")
    ma, _ = strategy_view(data, ("SMA", 60, 2), "min")
    _, x = observations(data, ma, FeatureConfig(1, 2, 2))
    assert x.loc[x.index < pd.Timestamp("2024-01-01 02:00"), "distance_to_ma"].isna().all()
    assert np.isfinite(x.loc["2024-01-01 02:00", "distance_to_ma"])


@pytest.mark.parametrize("candle", [1, 2, 3])
def test_daily_session_groups_available_only_after_final_session(candle):
    data = market(n=150)
    data.index = pd.date_range("2023-01-02", periods=len(data), freq="D")
    ma, _ = strategy_view(data, ("SMA", candle, 3), "d")
    _, full = observations(data, ma, FeatureConfig(2, 4, 4))
    prefix = data.iloc[:71]
    prefix_ma, _ = strategy_view(prefix, ("SMA", candle, 3), "d")
    _, partial = observations(prefix, prefix_ma, FeatureConfig(2, 4, 4))
    assert_frame_equal(full.loc[partial.index], partial)
    first_valid_time = data.index[candle*3-1] + pd.Timedelta(days=1)
    assert full.loc[full.index < first_valid_time, "distance_to_ma"].isna().all()


def test_missing_market_observation_not_backfilled():
    data = market().drop(market().index[200])
    ma, _ = strategy_view(data, ("SMA", 1, 3))
    p, x = observations(data, ma)
    missing = pd.Timestamp("2023-01-10 09:00")
    assert pd.isna(p.loc[missing])
    assert x.loc[missing].isna().all()


def test_original_signals_disabled_equality_and_pnl(manual_filter):
    data = market()
    bid, ask, mid = prepared_ohlc(data, 2, "h")
    legacy = main("SMA", mid.close, 4, 2, data, bid_df=bid, ask_df=ask)
    _, signals = strategy_view(data, ("SMA", 2, 4), weekly_closes=True)
    expected = legacy.copy()
    expected.index += pd.Timedelta(hours=2)
    assert_frame_equal(signals, expected)
    p = pd.DataFrame(.5, index=signals.index, columns=[0, 1])
    disabled, decisions = gate_signals(signals, p, manual_filter, use_markov_filter=False)
    assert_frame_equal(disabled, signals)
    assert decisions.allow_long_entry.all()
    trades = trades_from_signals(disabled)
    original_pnl = legacy.Prices.diff().loc[legacy.Signals == -1]
    assert_allclose(trades.pnl, original_pnl)
    assert_allclose(backtest(legacy, ask.low)[0], (trades.pnl > 0).mean())


def test_gate_preserves_exits_no_shorts_no_invented_orders(manual_filter):
    data = market()
    ma, signals = strategy_view(data, ("SMA", 2, 4), weekly_closes=True)
    p = pd.DataFrame(.01, index=signals.index, columns=[0, 1])
    p[0], p[1] = .99, .01
    p.iloc[::4] = [.01, .99]  # reject half of entries
    # Every exit's regime can be unfavorable; exit still unchanged.
    p.loc[signals.Signals == -1] = [.01, .99]
    selected, decisions = gate_signals(signals, p, manual_filter, use_markov_filter=True)
    original = trades_from_signals(signals).set_index("entry_time")
    chosen = trades_from_signals(selected).set_index("entry_time")
    assert_frame_equal(chosen, original.loc[chosen.index])
    assert len(chosen) > 0 and len(chosen) < len(original)
    assert selected.Signals.tolist() == [1, -1]*len(chosen)
    assert len(decisions) == len(original)
    empty, _ = gate_signals(signals.iloc[:0], p, manual_filter, use_markov_filter=True)
    assert empty.empty


@pytest.mark.parametrize("reason", ["not_converged", "invalid_covariance", "not_fitted"])
def test_invalid_model_blocks_entries(manual_filter, reason):
    manual_filter.valid, manual_filter.reason = False, reason
    _, signals = strategy_view(market(), ("SMA", 1, 3), weekly_closes=True)
    p = pd.DataFrame([[1., 0.]]*len(signals), index=signals.index)
    chosen, decisions = gate_signals(signals, p, manual_filter, use_markov_filter=True)
    assert chosen.empty
    assert set(decisions.reason) == {reason}


def test_no_favorable_state_and_invalid_probability_block(manual_filter):
    manual_filter.favorable_states.clear()
    assert not manual_filter.allow_long_entry([1., 0.])
    manual_filter.favorable_states = {0}
    for p in ([np.nan, np.nan], [-1, 2], [.5, .7]):
        assert not manual_filter.allow_long_entry(p)


def synthetic_states():
    rng = np.random.default_rng(100)
    states = np.repeat(np.tile([0, 1, 2], 8), 50)
    centers = np.array([[-4, -3, -2, 0, 3], [0, 3, 4, -4, 0], [4, 0, -4, 4, -3]])
    x = centers[states] + rng.normal(0, .3, (len(states), 5))
    return pd.DataFrame(x, columns=FEATURES, index=pd.date_range("2020-01-01", periods=len(x), freq="h")), states


def test_synthetic_regimes_reproducible_and_permutation_invariant():
    x, states = synthetic_states()
    a = MarkovRegimeFilter().fit(x, available_by=x.index[-1])
    b = MarkovRegimeFilter().fit(x, available_by=x.index[-1])
    assert a.valid, a.reason
    assert b.valid, b.reason
    pa, pb = a.predict_filtered_proba(x), b.predict_filtered_proba(x)
    assert_allclose(pa, pb, atol=1e-12)
    assert adjusted_rand_score(states, pa.to_numpy().argmax(axis=1)) > .9
    assert not a.favorable_states  # statistical separation is not profitability


def test_fit_guards_nonconvergence_scaler_and_cutoff():
    x, _ = synthetic_states()
    with pytest.raises(ValueError, match="cutoff"):
        MarkovRegimeFilter().fit(x, available_by=x.index[100])
    f = MarkovRegimeFilter(replace(MarkovConfig(), n_iter=1)).fit(x, available_by=x.index[-1])
    assert not f.valid and f.reason == "not_converged"
    a = MarkovRegimeFilter().fit(x.iloc[:600], available_by=x.index[599])
    assert_allclose(a.scaler.mean_, x.iloc[:600].mean())
    assert not MarkovRegimeFilter().fit(x.iloc[:10], available_by=x.index[10]).valid
    degenerate = MarkovRegimeFilter(replace(MarkovConfig(), min_state_observations=1000))
    degenerate.fit(x, available_by=x.index[-1])
    assert not degenerate.valid and degenerate.reason == "underpopulated_state"


def calibration_fixture(returns):
    start = pd.Timestamp("2024-01-01")
    times = pd.date_range(start+pd.Timedelta(hours=1), periods=len(returns), freq="3h")
    trades = pd.DataFrame({"entry_time": times, "exit_time": times+pd.Timedelta(hours=1),
                           "entry_price": 100., "exit_price": 100*(1+np.asarray(returns)),
                           "net_return": returns, "pnl": 100*np.asarray(returns), "holding_hours": 1.})
    p = pd.DataFrame([[1., 0.]]*len(times), index=times)
    return start, trades.exit_time.max(), trades, p


def test_small_sample_never_favorable(manual_filter):
    start, end, trades, p = calibration_fixture([.1]*29)
    summary, _ = calibrate(manual_filter, trades, p, start=start, end=end)
    assert not manual_filter.favorable_states
    assert set(summary.selection_reason) == {"insufficient_trades"}


def test_no_economic_edge_control(manual_filter):
    # Exactly balanced deterministic returns; high state confidence is not an edge.
    start, end, trades, p = calibration_fixture([.01, -.01]*100)
    summary, _ = calibrate(manual_filter, trades, p, start=start, end=end)
    assert not manual_filter.favorable_states
    assert summary.loc[0, "lower_mean_bound"] < 0


def test_positive_evidence_and_boundary_purge(manual_filter):
    start, end, trades, p = calibration_fixture([.01]*40)
    # A loss whose outcome is known only after calibration must be excluded.
    trades.loc[0, "exit_time"] = end + pd.Timedelta(days=1)
    trades.loc[0, "net_return"] = -100
    trades.loc[1, "entry_time"] = start-pd.Timedelta(days=1)
    summary, used = calibrate(manual_filter, trades, p, start=start, end=end)
    assert len(used) == 38
    assert manual_filter.favorable_states == {0}
    assert summary.loc[0, "lower_mean_bound"] > 0
    assert contained_trades(trades, start, end).exit_time.max() <= end


def test_accounting_costs_drawdown_and_no_exposure_when_blocked():
    times = pd.date_range("2024-01-01", periods=4, freq="h")
    price = pd.Series([100., 100., 90., 110.], index=times)
    signals = pd.DataFrame({"Signals": [1, -1], "Prices": [100., 110.]}, index=times[[1, 3]])
    trades = trades_from_signals(signals, units=2, fee_bps=10)
    curve = equity_curve(price, trades, initial_capital=1000, units=2, fee_bps=10)
    assert_allclose(curve.equity.iloc[-1]-1000, trades.pnl.sum())
    assert curve.units.min() == 0 and curve.units.max() == 2
    decisions = pd.DataFrame({"allow_long_entry": [True], "p_favorable": [.8], "current_regime": [0]})
    metrics = comparison_metrics("test", curve, trades, decisions, 1000)
    assert metrics["Max Drawdown"] < -.02
    empty = equity_curve(price, trades.iloc[:0], initial_capital=1000)
    assert_allclose(empty.equity, 1000)


def test_persistence_schema_and_global_restore(tmp_path, manual_filter):
    path = tmp_path / "model.pkl"
    manual_filter.save(path)
    restored = MarkovRegimeFilter.load(path)
    assert_frame_equal(manual_filter.predict_filtered_proba(feature_frame()), restored.predict_filtered_proba(feature_frame()))
    with pytest.raises(ValueError, match="order"):
        restored.predict_filtered_proba(feature_frame().iloc[:, ::-1])
    original = keys.temp, keys.calls, keys.methods, keys.mid_cache, keys.fill_ohlc_dict
    with pytest.raises(RuntimeError):
        with optimizer_settings(RunConfig()):
            keys.fill_ohlc_dict(market())
            assert 5 in keys.mid_cache
            raise RuntimeError()
    assert (keys.temp, keys.calls, keys.methods, keys.mid_cache, keys.fill_ohlc_dict) == original


def test_nonpositive_market_refused():
    data = market()
    data.iloc[10, 3] = 0
    with pytest.raises(ValueError, match="Nonpositive"):
        validate_market(data)


def test_daily_sharpe_uses_closed_days_without_extra_initial_zero():
    index = pd.date_range("2024-01-01", periods=4, freq="D")
    curve = pd.DataFrame({"equity": [100., 110., 99., 108.9], "units": 0.}, index=index)
    trades = trades_from_signals(pd.DataFrame({"Signals": [], "Prices": []}, index=index[:0]))
    decisions = pd.DataFrame(columns=["allow_long_entry", "current_regime", "p_favorable"])
    result = comparison_metrics("test", curve, trades, decisions, 100.)
    returns = np.array([.1, -.1, .1])
    assert_allclose(result["Sharpe Ratio"], returns.mean()/returns.std(ddof=1)*np.sqrt(365.25))
