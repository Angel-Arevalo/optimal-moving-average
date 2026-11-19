import argparse
from pathlib import Path
import pandas as pd
from .config import MarkovConfig, RunConfig
from .walk_forward import ROOT, run_walk_forward


def main():
    parser = argparse.ArgumentParser(description="Original versus causal entry-only HMM, weekly walk-forward")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "BTC_hourly.parquet")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--weeks", type=int, default=8)
    parser.add_argument("--optimization-weeks", type=int, default=26)
    parser.add_argument("--fit-weeks", type=int, default=26)
    parser.add_argument("--calibration-weeks", type=int, default=52)
    parser.add_argument("--calls", type=int, default=20)
    parser.add_argument("--candle-max", type=int, default=6)
    parser.add_argument("--lookback-max", type=int, default=40)
    parser.add_argument("--temporality", choices=["min", "h", "d"], default="h")
    parser.add_argument("--n-states", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=.70)
    parser.add_argument("--min-trades", type=int, default=30)
    parser.add_argument("--capital", type=float, default=100000.)
    parser.add_argument("--units", type=float, default=1.)
    parser.add_argument("--fee-bps", type=float, default=0.)
    parser.add_argument("--experiments", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "btc_ab")
    args = parser.parse_args()
    config = RunConfig(optimization_weeks=args.optimization_weeks, fit_weeks=args.fit_weeks,
                       calibration_weeks=args.calibration_weeks, evaluation_weeks=args.weeks,
                       optimizer_calls=args.calls, candle_max=args.candle_max,
                       lookback_max=args.lookback_max, temporality=args.temporality,
                       initial_capital=args.capital, units=args.units, fee_bps=args.fee_bps)
    markov = MarkovConfig(n_components=args.n_states, probability_threshold=args.threshold,
                          min_trades_per_regime=args.min_trades)
    result = run_walk_forward(pd.read_parquet(args.data), args.start, config, markov,
                              output=args.output, experiments=args.experiments)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
