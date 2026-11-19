"""Entry gate, original trade pairing, economic calibration and reporting."""
import numpy as np
import pandas as pd
from .config import FEATURES

TRADE_COLUMNS = ["entry_time", "entry_price", "exit_time", "exit_price", "pnl",
                 "net_return", "holding_hours"]


def trades_from_signals(signals, *, units=1., fee_bps=0.):
    if len(signals) % 2 or not np.array_equal(signals.Signals.to_numpy(), np.tile([1, -1], len(signals) // 2)):
        raise ValueError("Expected alternating complete original LONG entry/exit pairs")
    rows = []
    for i in range(0, len(signals), 2):
        entry, exit_ = signals.iloc[i:i+2].Prices.to_numpy()
        start, end = signals.index[i:i+2]
        if end <= start or min(entry, exit_) <= 0:
            raise ValueError("Invalid original trade")
        pnl = units * ((exit_ - entry) - (entry + exit_) * fee_bps / 10000)
        rows.append(dict(entry_time=start, entry_price=entry, exit_time=end, exit_price=exit_,
                         pnl=pnl, net_return=pnl/(units*entry), holding_hours=(end-start).total_seconds()/3600))
    return pd.DataFrame(rows, columns=TRADE_COLUMNS)


def contained_trades(trades, start, end):
    """Purge boundary-crossing trades; only fully known labels in (start,end]."""
    return trades.loc[(trades.entry_time > start) & (trades.exit_time <= end)].copy()


def gate_signals(signals, probabilities, regime_filter, *, use_markov_filter=None):
    """Only retain original pairs. No new entries, price changes or exit decisions."""
    trades_from_signals(signals)  # validate pair invariant
    if use_markov_filter is None:
        use_markov_filter = regime_filter.config.enabled
    keep, decisions = [], []
    for i in range(0, len(signals), 2):
        time = signals.index[i]
        p = (probabilities.loc[time].to_numpy() if time in probabilities.index
             else np.full(regime_filter.config.n_components, np.nan))
        decision = regime_filter.decision(p)
        if not use_markov_filter:
            decision.update(allow_long_entry=True, reason="filter_disabled")
        elif not regime_filter.valid:
            decision.update(allow_long_entry=False, reason=regime_filter.reason)
        decision.update(entry_time=time)
        decision.update({f"p_{j}": float(value) for j, value in enumerate(p)})
        decisions.append(decision)
        if decision["allow_long_entry"]:
            keep.extend([i, i+1])
    cols = ["allow_long_entry", "current_regime", "regime_confidence", "p_favorable", "reason", "entry_time"]
    cols += [f"p_{j}" for j in range(regime_filter.config.n_components)]
    return signals.iloc[keep].copy(), pd.DataFrame(decisions, columns=cols)


def annotate_trades(trades, probabilities):
    result = trades.copy()
    p = probabilities.reindex(pd.DatetimeIndex(result.entry_time)).to_numpy()
    valid = np.isfinite(p).all(axis=1)
    result["current_regime"] = np.where(valid, np.nan_to_num(p).argmax(axis=1), -1)
    for j in range(probabilities.shape[1]):
        result[f"p_{j}"] = p[:, j]
    return result


def profit_factor(returns):
    gain = returns[returns > 0].sum()
    loss = -returns[returns < 0].sum()
    return float(gain / loss) if loss else (np.inf if gain else np.nan)


def trade_statistics(trades):
    r = trades.net_return
    return dict(number_of_trades=len(r), win_rate=float((r > 0).mean()) if len(r) else np.nan,
                average_net_return=r.mean(), median_net_return=r.median(),
                profit_factor=profit_factor(r), average_holding_time=trades.holding_hours.mean())


def block_lower_bound(returns, config):
    """Circular moving blocks of consecutive trades assigned to this state.

    One-sided Bonferroni-adjusted bound across the fitted model's states.
    """
    r = np.asarray(returns, dtype=float)
    if len(r) < config.min_trades_per_regime or not np.isfinite(r).all():
        return np.nan
    rng = np.random.default_rng(config.random_state)
    block = min(config.bootstrap_block, len(r))
    starts = rng.integers(0, len(r), size=(config.bootstrap_samples, int(np.ceil(len(r)/block))))
    indices = ((starts[..., None] + np.arange(block)) % len(r)).reshape(config.bootstrap_samples, -1)[:, :len(r)]
    means = r[indices].mean(axis=1)
    return float(np.quantile(means, (1-config.confidence)/config.n_components))


def calibrate(regime_filter, trades, probabilities, *, start, end):
    if regime_filter.fit_end is not None and start < regime_filter.fit_end:
        raise ValueError("Calibration must follow HMM fitting")
    if len(probabilities) and (probabilities.index.min() <= start or probabilities.index.max() > end):
        raise ValueError("Calibration probabilities outside calibration block")
    clean = annotate_trades(contained_trades(trades, start, end), probabilities)
    regime_filter.favorable_states = set()
    rows = []
    for state in range(regime_filter.config.n_components):
        subset = clean.loc[clean.current_regime == state].sort_values("entry_time")
        stats = trade_statistics(subset)
        lower = block_lower_bound(subset.net_return, regime_filter.config)
        enough = len(subset) >= regime_filter.config.min_trades_per_regime
        favorable = bool(regime_filter.valid and enough and np.isfinite(lower) and lower > 0
                         and stats["average_net_return"] > 0)
        if favorable:
            regime_filter.favorable_states.add(state)
        reason = (regime_filter.reason if not regime_filter.valid else
                  "insufficient_trades" if not enough else
                  "positive_lower_bound" if favorable else "nonpositive_lower_bound")
        rows.append(dict(regime=state, **stats, lower_mean_bound=lower, favorable=favorable, selection_reason=reason))
    regime_filter.calibration_end = pd.Timestamp(end)
    if regime_filter.valid:
        regime_filter.reason = "ready" if regime_filter.favorable_states else "no_favorable_states"
    return pd.DataFrame(rows), clean


def state_summary(features, probabilities, trades, price, regime_filter):
    """Descriptive only. Forward return stays within the supplied phase."""
    finite = probabilities.notna().all(axis=1)
    states = probabilities.fillna(-1).to_numpy().argmax(axis=1)
    assigned = pd.Series(np.where(finite, states, -1), index=features.index)
    runs = assigned.ne(assigned.shift()).cumsum()
    run_table = pd.DataFrame({"state": assigned, "run": runs}).groupby("run").agg(state=("state", "first"), length=("state", "size"))
    future = price.shift(-1) / price - 1  # diagnostic only; NEVER an HMM observation
    enriched = trades if "current_regime" in trades else annotate_trades(trades, probabilities)
    rows = []
    for state in range(regime_filter.config.n_components):
        mask = assigned == state
        subset = enriched.loc[enriched.current_regime == state]
        rows.append(dict(regime=state, frequency=float(mask.mean()) if len(mask) else np.nan,
                         mean_duration_observations=run_table.loc[run_table.state == state, "length"].mean(),
                         persistence_probability=regime_filter.model.transmat_[state, state] if regime_filter.valid else np.nan,
                         mean_state_probability=probabilities[state].mean(),
                         future_return_1_mean=future.reindex(features.index)[mask].mean(),
                         number_of_signals=len(subset), **trade_statistics(subset),
                         **{f"mean_{name}": features.loc[mask, name].mean() for name in FEATURES}))
    return pd.DataFrame(rows)


def equity_curve(price, trades, *, initial_capital=100000., units=1., fee_bps=0., open_positions=None):
    """Close mark-to-market, fixed units. No implicit compounding/size optimization."""
    cash_events = pd.Series(0., index=price.index)
    quantity_events = cash_events.copy()
    for tr in trades.itertuples():
        if tr.entry_time not in price.index or tr.exit_time not in price.index:
            raise ValueError("Trade timestamps absent from valuation grid")
        cash_events.loc[tr.entry_time] -= units * tr.entry_price * (1+fee_bps/10000)
        cash_events.loc[tr.exit_time] += units * tr.exit_price * (1-fee_bps/10000)
        quantity_events.loc[tr.entry_time] += units
        quantity_events.loc[tr.exit_time] -= units
    if open_positions is not None:
        for position in open_positions.itertuples():
            if position.entry_time not in price.index:
                raise ValueError("Open entry absent from valuation grid")
            cash_events.loc[position.entry_time] -= units * position.entry_price * (1+fee_bps/10000)
            quantity_events.loc[position.entry_time] += units
    quantity = quantity_events.cumsum()
    if (quantity < 0).any() or (quantity > units * (1+1e-12)).any():
        raise ValueError("SHORT or overlapping positions detected")
    marked_position = (quantity * price.ffill()).where(quantity.ne(0), 0.)
    equity = initial_capital + cash_events.cumsum() + marked_position
    return pd.DataFrame({"equity": equity, "units": quantity}, index=price.index)


def comparison_metrics(strategy, curve, trades, decisions, initial_capital):
    equity = curve.equity
    total = equity.iloc[-1] / initial_capital - 1
    years = (equity.index[-1] - equity.index[0]).total_seconds() / (365.25*86400)
    annualized = (equity.iloc[-1]/initial_capital)**(1/years)-1 if years > 0 and (equity > 0).all() else np.nan
    peak = equity.cummax().clip(lower=initial_capital)
    # Availability timestamps denote closes: 00:00 closes the previous day.
    # A first midnight is the initial valuation, not an extra zero-return day.
    daily = equity.resample("1D", closed="right", label="right").last().ffill()
    daily_returns = daily.pct_change()
    if equity.index[0] == equity.index[0].normalize():
        daily_returns = daily_returns.iloc[1:]
    else:
        daily_returns.iloc[0] = daily.iloc[0] / initial_capital - 1
    sharpe = (daily_returns.mean()/daily_returns.std(ddof=1)*np.sqrt(365.25)
              if len(daily_returns) > 1 and daily_returns.std(ddof=1) > 0 and (equity > 0).all() else np.nan)
    n = len(decisions)
    accepted = int(decisions.allow_long_entry.sum())
    active = decisions.loc[decisions.allow_long_entry]
    dt = curve.index.to_series().diff().dt.total_seconds().fillna(0)
    exposure = float((dt * curve.units.shift(1).fillna(0).gt(0)).sum()/dt.sum()) if dt.sum() else 0.
    return {"Strategy": strategy, "Total Return": total, "Annualized Return": annualized,
            "Max Drawdown": float((equity/peak-1).min()), "Sharpe Ratio": sharpe,
            "Profit Factor": profit_factor(trades.pnl), "Win Rate": (trades.net_return > 0).mean(),
            "Number of Trades": len(trades), "Market Exposure": exposure,
            "Average Trade Return": trades.net_return.mean(), "Original Entry Signals": n,
            "Accepted Entry Signals": accepted, "Rejected Entry Signals": n-accepted,
            "Acceptance Rate": accepted/n if n else np.nan,
            "Average P(favorable) At Entry": active.p_favorable.mean(),
            "Filtered Regime Distribution": str(active.current_regime.value_counts().to_dict()),
            "Net PnL": equity.iloc[-1]-initial_capital,
            "Closed Trade PnL": trades.pnl.sum(),
            "Open Position PnL": equity.iloc[-1]-initial_capital-trades.pnl.sum(),
            "Open Positions": int(curve.units.iloc[-1] > 0)}
