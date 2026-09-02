from typing import List, Tuple, Dict, Any
import copy
import numpy as np
import optuna
import pandas as pd

from direction_methods import dir_main
import keys

from tester import backtest
from find_best import f
from use_tecnics import main

dir_methods = [    
    "KEF",
    "HURST",
    "LO_MACKINLAY",
    "ADX",
    "VOL_RATIO",
    "SHANNON",
    "ATR_EXPANSION",
    "KELTNER_BREAKOUT",
    "OU_REVERSION"
]

optuna.logging.set_verbosity(optuna.logging.WARNING)


def make_cache_key(ma_method: str, ma_candle: int, ma_lookback: int, params: dict, filter_val: bool) -> tuple:
    sorted_params = tuple(sorted((k, v) for k, v in params.items()))
    return (ma_method, ma_candle, ma_lookback, filter_val, sorted_params)


def generate_deterministic_neighbors(base_params: dict) -> list[dict]:
    neighbors = [base_params]

    OFFSETS = [
        {"int": 1, "float": 0.05},
        {"int": -1, "float": -0.05},
        {"int": 2, "float": 0.10},
        {"int": -2, "float": -0.10},
        {"int": 1, "float": -0.05},
        {"int": -1, "float": 0.05},
        {"int": 3, "float": 0.15},
        {"int": -3, "float": -0.15},
    ]

    BOUNDS = {
        "candle": (1, 100, int),
        "window": (2, 100, int),
        "follow_tend": (0.01, 50.0, float),
        "k": (2, 10, int),
        "bins": (5, 20, int),
        "lookback_ma": (20, 100, int),
        "mult": (1.0, 4.0, float),
    }

    for offset in OFFSETS:
        neighbor = copy.deepcopy(base_params)
        for key, val in base_params.items():
            if key in BOUNDS:
                min_v, max_v, v_type = BOUNDS[key]
                if v_type == int:
                    new_val = int(np.clip(val + offset["int"], min_v, max_v))
                else:
                    new_val = float(np.clip(val * (1.0 + offset["float"]), min_v, max_v))
                neighbor[key] = new_val
        neighbors.append(neighbor)

    return neighbors


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

        filter = trial.suggest_categorical(
            "side", ["both", "revert", "tend"]
        )

        ohlc: pd.DataFrame = keys.mid_cache[ma_candle]["close"]
        signals_prices: pd.DataFrame = main(
            ma_method, ohlc, ma_lookback, ma_candle, asset, shorts
        )

        ht, rr, pr, tr, mae, sqn = backtest(signals_prices, ohlc, True, shorts)

        dir_method = trial.suggest_categorical("name", dir_methods)

        # NO USAR POR EL MOMENTO, BACK_DIR NO LO 
        # SOPORTA
        if dir_method == "":
            t = f(ht, rr, pr, tr, sqn, mae)
            trial.set_user_attr("hr", ht)
            trial.set_user_attr("rr", rr)
            trial.set_user_attr("pr", pr)
            trial.set_user_attr("tr", tr)
            trial.set_user_attr("mae", mae)
            trial.set_user_attr("sqn", sqn)
            trial.set_user_attr("p_score", t)
            trial.set_user_attr("std_score", 0)
            trial.set_user_attr("side", filter)

            return t

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

        hr_dir, rr_dir, pr_dir, tr_dir, mae_dir, sqn_dir = dir_main(signals_prices, asset, ma_candle, params, shorts, filter)

        trial.set_user_attr("hr", hr_dir)
        trial.set_user_attr("rr", rr_dir)
        trial.set_user_attr("pr", pr_dir)
        trial.set_user_attr("tr", tr_dir)
        trial.set_user_attr("mae", mae_dir)
        trial.set_user_attr("sqn", sqn_dir)
        trial.set_user_attr("side", filter)

        return -f(hr_dir, rr_dir, pr_dir, tr_dir, sqn_dir, mae_dir)

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
        print(f"Operando {'cortos' if shorts else 'largos'} con filtro {best_metrics["filter"]}\n")

    return best_params, best_metrics
