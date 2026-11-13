import argparse
from find_best import opti_main
from find_best_dir import opti_dir
import keys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("--asset")
    parser.add_argument("--shorts", action="store_true")
    parser.add_argument("--direction", action="store_true")
    parser.add_argument("--filter", action="store_true")
    parser.add_argument("--improvement-weight", type=float, default=.75)
    parser.add_argument("--calls", type=int, default=60)
    parser.add_argument("--sessions", type=int, default=1)
    parser.add_argument("--temporality", choices=["auto", "min", "h", "d"], default="auto")
    args = parser.parse_args()
    keys.calls, keys.candles = args.calls, args.sessions
    keys.temp = args.temporality
    options = {"asset": args.asset, "shorts": args.shorts}
    if args.direction:
        print(opti_dir(args.data, n_trials=args.calls, filter=args.filter, improvement_weight=args.improvement_weight, **options))
    else:
        print(opti_main(args.data, **options))


if __name__ == "__main__":
    main()
