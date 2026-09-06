from typing import List, Tuple, Dict, Any
import copy
import numpy as np
import optuna
import pandas as pd

from direction_methods import dir_main
import keys

from tester import backtest
from find_best import softplus, f as f_single
from use_tecnics import main

dir_methods = [
    "KEF",
    "HURST",
#   "LO_MACKINLAY",
    "ADX",
#   "VOL_RATIO",
#   "SHANNON",
    "ATR_EXPANSION",
#   "KELTNER_BREAKOUT",
    "OU_REVERSION"
]

optuna.logging.set_verbosity(optuna.logging.WARNING)

def opti_dir(asset: pd.DataFrame, verbose: bool = True, shorts: bool = False, n_trials: int = 100) -> Tuple:
    keys.bid_cache = {}
    keys.ask_cache = {}
    keys.mid_cache = {}

    keys.fill_ohlc_dict(asset)

    completed_scores: List[float] = []
    score_cache: Dict[tuple, tuple] = {}

    def objective(trial: optuna.Trial) -> float:
        ma_method = trial.suggest_categorical(
            "ma_method", list(keys.methods)
        )
        ma_candle = trial.suggest_int(
            "ma_candle", keys.candles_min, keys.candles
        )
        ma_lookback = trial.suggest_int(
            "ma_lookback", keys.lookbacks_min, keys.lookbacks
        )

        filter_side = "revert" 

        ohlc: pd.DataFrame = keys.mid_cache[ma_candle]["close"]
        signals_prices: pd.DataFrame = main(ma_method, ohlc, ma_lookback, ma_candle, asset, shorts)

        if shorts:
            initial_params: tuple = backtest(signals_prices, keys.bid_cache[ma_candle]["high"], True, shorts)
        else:
            initial_params: tuple = backtest(signals_prices, keys.ask_cache[ma_candle]["low"], True, shorts)

        dir_method = trial.suggest_categorical("name", dir_methods)
        dir_candle = trial.suggest_int("candle", 1, 100)
        dir_window = trial.suggest_int("window", 2, 100)

        params = {
            "name": dir_method,
            "candle": dir_candle,
            "window": dir_window,
        }

        if dir_method == "KEF":
            params["follow_tend"] = trial.suggest_float("follow_tend", 0.1, 1.0)
        elif dir_method == "HURST":
            params["follow_tend"] = trial.suggest_float("follow_tend", 0.5, 1.0)
        elif dir_method == "LO_MACKINLAY":
            params["k"] = trial.suggest_int("k", 2, 10)
            params["follow_tend"] = trial.suggest_float(
                "follow_tend", 0.5, 3.0
            )
        elif dir_method == "ADX":
            params["follow_tend"] = trial.suggest_float(
                "follow_tend", 10.0, 40.0
            )
        elif dir_method == "VOL_RATIO":
            params["follow_tend"] = trial.suggest_float(
                "follow_tend", 0.5, 3.0
            )
        elif dir_method == "SHANNON":
            params["bins"] = trial.suggest_int("bins", 5, 20)
            params["follow_tend"] = trial.suggest_float(
                "follow_tend", 0.3, 1.0
            )
        elif dir_method == "ATR_EXPANSION":
            params["lookback_ma"] = trial.suggest_int("lookback_ma", 20, 100)
            params["follow_tend"] = trial.suggest_float(
                "follow_tend", 1.0, 3.0
            )
        elif dir_method == "KELTNER_BREAKOUT":
            params["mult"] = trial.suggest_float("mult", 1.0, 4.0)
        elif dir_method == "OU_REVERSION":
            params["follow_tend"] = trial.suggest_float("follow_tend", 0.01, 1.0)

        filter_params: tuple = dir_main(signals_prices, asset, ma_candle, params, shorts, filter_side)

        hr_dir, rr_dir, pr_dir, tr_dir, mae_dir, sqn_dir = filter_params

        trial.set_user_attr("hr", hr_dir)
        trial.set_user_attr("rr", rr_dir)
        trial.set_user_attr("pr", pr_dir)
        trial.set_user_attr("tr", tr_dir)
        trial.set_user_attr("mae", mae_dir)
        trial.set_user_attr("sqn", sqn_dir)
        trial.set_user_attr("side", filter_side)
        trial.set_user_attr("hr_", initial_params[0])
        trial.set_user_attr("rr_", initial_params[1])
        trial.set_user_attr("pr_", initial_params[2])
        trial.set_user_attr("tr_", initial_params[3])
        trial.set_user_attr("mae_", initial_params[4])
        trial.set_user_attr("sqn_", initial_params[5])

        return -f(initial_params, filter_params)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(),
    )

    study.optimize(objective, n_trials=n_trials)

    best_trial = study.best_trial

    best_params: Dict[str, Any] = best_trial.params
    best_metrics: Dict[str, float] = {
        "score": -best_trial.value,
        "hr": best_trial.user_attrs.get("hr"),
        "rr": best_trial.user_attrs.get("rr"),
        "pr": best_trial.user_attrs.get("pr"),
        "tr": best_trial.user_attrs.get("tr"),
        "mae": best_trial.user_attrs.get("mae"),
        "sqn": best_trial.user_attrs.get("sqn"),
        "filter": best_trial.user_attrs.get("side"),
        "hr_": best_trial.user_attrs.get("hr_"),
        "rr_": best_trial.user_attrs.get("rr_"),
        "pr_": best_trial.user_attrs.get("pr_"),
        "tr_": best_trial.user_attrs.get("tr_"),
        "mae_": best_trial.user_attrs.get("mae_"),
        "sqn_": best_trial.user_attrs.get("sqn_"),

    }

    if verbose:
        hr_0  = best_metrics.get('hr_') or 0.0
        hr_f  = best_metrics.get('hr') or 0.0
        rr_0  = best_metrics.get('rr_') or 0.0
        rr_f  = best_metrics.get('rr') or 0.0
        pr_0  = best_metrics.get('pr_') or 0.0
        pr_f  = best_metrics.get('pr') or 0.0
        tr_0  = best_metrics.get("tr_") if best_metrics.get("tr_") is not None else "N/A" 
        tr_f  = best_metrics.get('tr') if best_metrics.get('tr') is not None else "N/A"
        sqn_0 = best_metrics.get('sqn_') or 0.0
        sqn_f = best_metrics.get('sqn') or 0.0
        mae_0 = best_metrics.get('mae_') or 0.0
        mae_f = best_metrics.get('mae') or 0.0

        print(f"\nMethod: {best_params.get('name')} | Side: {'Shorts' if shorts else 'Longs'} | Filter: {best_metrics['filter']}")
        print(f"Params: {best_params}\n")

        print(f"{'Métrica':<15} | {'Base (Inicial)':<15} | {'Filtrado':<15}")
        print("-" * 49)

        print(f"{'Hit Ratio':<15} | {hr_0 * 100:>14.2f}% | {hr_f * 100:>14.2f}%")
        print(f"{'Risk Reward':<15} | {rr_0:>15.4f} | {rr_f:>15.4f}")
        print(f"{'Profit Factor':<15} | {pr_0:>15.4f} | {pr_f:>15.4f}")
        print(f"{'Trades':<15} | {str(tr_0):>15} | {str(tr_f):>15}")
        print(f"{'SQN':<15} | {sqn_0:>15.4f} | {sqn_f:>15.4f}")
        print(f"{'MAE':<15} | {mae_0:>15.5f} | {mae_f:>15.5f}")
        print("-" * 49 + "\n")

    return best_params, best_metrics

def f(initial_params: Tuple[float, ...], filter_params: Tuple[float, ...]) -> float:
    hr_0, rr_0, pr_0, tr_0, mae_0, sqn_0 = initial_params
    hr_f, rr_f, pr_f, tr_f, mae_f, sqn_f = filter_params

    eps = 1e-6

    q_base = f_single(hr_0, rr_0, pr_0, int(tr_0), sqn_0, mae_0)

    q_filtered = f_single(hr_f, rr_f, pr_f, int(tr_f), sqn_f, mae_f)

    r_improvement = (q_filtered + eps) / (q_base + eps)

    d_hr = hr_f - hr_0
    k_gate = 100.0

    gate_hr = 1.0 / (1.0 + np.exp(-k_gate * d_hr))

    final_score = q_filtered * r_improvement * gate_hr

    return float(final_score)
