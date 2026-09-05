from typing import List, Tuple, Dict, Any
import copy
import numpy as np
import optuna
import pandas as pd

from direction_methods import dir_main
import keys

from tester import backtest
from find_best import softplus
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

        filter_side = "both" 

        ohlc: pd.DataFrame = keys.mid_cache[ma_candle]["close"]
        signals_prices: pd.DataFrame = main(
            ma_method, ohlc, ma_lookback, ma_candle, asset, shorts
        )

        initial_params: tuple = backtest(signals_prices, ohlc, True, shorts)

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
        "filter": best_trial.user_attrs.get("side")
    }

    if verbose:
        print(f"Método de Dirección: {best_params.get('name')}, Datos optimizados {best_params}")
        print(f"\nhit ratio: {best_metrics['hr']}")
        print(f"risk reward: {best_metrics['rr']}")
        print(f"profit factor: {best_metrics['pr']}")
        print(f"trades: {best_metrics['tr']}")
        print(f"Sqn: {best_metrics['sqn']:.4f}")
        print(f"Mae: {best_metrics['mae']}")
        print(f"Operando {'cortos' if shorts else 'largos'} con filtro {best_metrics['filter']}\n")

    return best_params, best_metrics


def f(
    initial_params: tuple[float, float, float, float, float, float], 
    filter_params: tuple[float, float, float, float, float, float],
    beta: float = 2.0,
    lmbda: float = 1.5
) -> float:
    e: float = 1e-6

    hr_0, rr_0, pf_0, tr_0, mae_0, sqn_0 = initial_params
    hr_f, rr_f, pf_f, tr_f, mae_f, sqn_f = filter_params

    utilidad = (
        np.log((pf_f + e) / (pf_0 + e)) +
        np.log((sqn_f + e) / (sqn_0 + e)) +
        np.log((rr_f + e) / (rr_0 + e)) +
        np.log((mae_0 + e) / (mae_f + e))
    )

    delta_hr_rel = np.log((hr_f + e) / (hr_0 + e))
    delta_hr_rel_n = delta_hr_rel * np.sqrt(tr_f)

    score = (
        utilidad + 
        np.log(1 + softplus(delta_hr_rel_n, beta)) - 
        lmbda * softplus(-delta_hr_rel_n, beta)
    )

    return float(score)
