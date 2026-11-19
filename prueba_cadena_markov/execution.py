"""Continuous reference LONG positions; HMM only accepts/rejects each entry."""
import numpy as np
import pandas as pd
from .economics import TRADE_COLUMNS


def completed_crossing_pairs(signals):
    """Historical labels: ignore orphan exits, retain only naturally closed pairs.

    A terminal entry is censored, never closed at the dataset boundary.
    """
    pairs, entry = [], None
    for i, signal in enumerate(signals.Signals):
        if signal == 1 and entry is None:
            entry = i
        elif signal == -1 and entry is not None:
            pairs.extend([entry, i])
            entry = None
    return signals.iloc[pairs].copy()


class ContinuousExecution:
    """One shared reference signal stream, identical candidate trades for A/B.

    A rejected reference position still defines the next original exit; HMM
    cannot insert extra trades while the reference strategy is occupied.
    Entry-time MA parameters and warmup origin remain frozen until that exit.
    """
    def __init__(self, *, units=1., fee_bps=0., n_components=3):
        self.units, self.fee_bps, self.n_components = units, fee_bps, n_components
        self.position = None
        self.closed = []
        self.decisions = {"Original": [], "HMM": []}
        self.last_end = None

    def advance(self, current_signals, carried_signals, probabilities, model,
                *, start, end, window, parameters, history_start):
        if self.last_end is not None and start != self.last_end:
            raise ValueError("Execution windows must be contiguous")
        current = current_signals.loc[(current_signals.index > start) & (current_signals.index <= end)]
        carried = carried_signals.loc[(carried_signals.index > start) & (carried_signals.index <= end)]
        using_carried = self.position is not None
        for time in current.index.union(carried.index).sort_values():
            if self.position is not None:
                exit_stream = carried if using_carried else current
                if time in exit_stream.index and exit_stream.loc[time, "Signals"] == -1:
                    position = self.position
                    exit_price = float(exit_stream.loc[time, "Prices"])
                    pnl = self.units * (exit_price-position["entry_price"]
                                       -(exit_price+position["entry_price"])*self.fee_bps/10000)
                    self.closed.append(dict(position, exit_time=time, exit_price=exit_price,
                                            exit_window=window, exit_reason="ma_cross", pnl=pnl,
                                            net_return=pnl/(self.units*position["entry_price"]),
                                            holding_hours=(time-position["entry_time"]).total_seconds()/3600))
                    self.position = None
                    using_carried = False
                # No same-timestamp reopening when a frozen old MA exits.
                continue
            if time not in current.index or current.loc[time, "Signals"] != 1:
                continue
            p = (probabilities.loc[time].to_numpy() if time in probabilities.index
                 else np.full(model.config.n_components, np.nan))
            decision = model.decision(p)
            if not model.valid:
                decision.update(allow_long_entry=False, reason=model.reason)
            decision.update(entry_time=time, window=window)
            decision.update({f"p_{j}": float(v) for j, v in enumerate(p)})
            self.decisions["HMM"].append(decision.copy())
            self.decisions["Original"].append(dict(decision, allow_long_entry=True, reason="filter_disabled"))
            self.position = dict(entry_time=time, entry_price=float(current.loc[time, "Prices"]),
                                 window=window, parameters=tuple(parameters), history_start=history_start,
                                 accepted=bool(decision["allow_long_entry"]),
                                 current_regime=decision["current_regime"],
                                 **{f"p_{j}": float(v) for j, v in enumerate(p)})
            using_carried = False
        self.last_end = end

    def trades(self, name):
        cols = TRADE_COLUMNS + ["window", "exit_window", "exit_reason", "parameters", "history_start",
                                "accepted", "current_regime"] + [f"p_{j}" for j in range(self.n_components)]
        rows = self.closed if name == "Original" else [r for r in self.closed if r["accepted"]]
        return pd.DataFrame(rows, columns=cols)

    def open_positions(self, name):
        rows = []
        if self.position is not None and (name == "Original" or self.position["accepted"]):
            rows = [self.position.copy()]
        cols = ["entry_time", "entry_price", "window", "parameters", "history_start", "accepted",
                "current_regime"] + [f"p_{j}" for j in range(self.n_components)]
        return pd.DataFrame(rows, columns=cols)

    def decision_frame(self, name):
        cols = ["entry_time", "window", "allow_long_entry", "reason", "current_regime",
                "regime_confidence", "p_favorable"] + [f"p_{j}" for j in range(self.n_components)]
        return pd.DataFrame(self.decisions[name], columns=cols)
