from typing import List, Tuple
import numpy as np
import optuna
import pandas as pd

from direction_methods import dir_main
from find_best import f
import keys

from tester import backtest
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
]

def calculate_directional_score(val: float, fsr: float, def_v: float, msr: float) -> float:
    s_fsr = 1.0 / (1.0 + np.exp(-3.0 * (fsr - 1.0)))

    s_def = 1.0 / (1.0 + np.exp(-3.0 * (def_v - 1.0)))

    s_dir = 1.0 + 0.4 * (s_fsr - 0.5) + 0.3 * (s_def - 0.5)

    msr_safe = max(msr, 1e-4)
    msr_penalty = 0.5 * max(0.0, np.abs(np.log(msr_safe)) - np.log(10.0)) ** 2

    final_score = (val * s_dir) - msr_penalty
    return float(final_score)


def opti_dir(asset: pd.DataFrame, verbose: bool = True, shorts: bool = False, n_trials: int = 100) -> optuna.Study:
    keys.bid_cache = {}
    keys.ask_cache = {}
    keys.mid_cache = {}

    keys.fill_ohlc_dict(asset)

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

        ohlc: pd.DataFrame = keys.mid_cache[ma_candle]["close"]
        signals_prices: pd.DataFrame = main(
            ma_method, ohlc, ma_lookback, shorts, asset
        )

        if shorts:
            bid_ask_data = keys.bid_cache[ma_candle]["high"]
        else:
            bid_ask_data = keys.ask_cache[ma_candle]["low"]

        hr, rr, pr, tr, mae = backtest(
            signals_prices, bid_ask_data, False, shorts
        )

        val: float = f(hr, rr, pr, tr, mae)

        dir_method = trial.suggest_categorical("dir_method", dir_methods)

        if dir_method == "":
            return -val

        dir_candle = trial.suggest_int("dir_candle", 1, 100)
        dir_window = trial.suggest_int("dir_window", 2, 100)

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

        # Aquí asumo que ya modificaste dir_main para que retorne hr, rr, pr, tr, mae
        hr_dir, rr_dir, pr_dir, tr_dir, mae_dir = dir_main(
            signals_prices, asset, ma_candle, params, shorts
        )

        # Guardamos los KPIs en el trial actual para poder recuperarlos después
        trial.set_user_attr("hr", hr_dir)
        trial.set_user_attr("rr", rr_dir)
        trial.set_user_attr("pr", pr_dir)
        trial.set_user_attr("tr", tr_dir)
        trial.set_user_attr("mae", mae_dir)

        return -f(hr_dir, rr_dir, pr_dir, tr_dir, mae_dir)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(),
    )

    study.optimize(objective, n_trials=n_trials)

    if verbose:
        best_trial = study.best_trial
        print(f"Método de Dirección: {best_trial.params.get('dir_method')}, Datos optimizados {best_trial.params}")
        print(f"\nhit ratio: {best_trial.user_attrs.get('hr')}")
        print(f"risk reward: {best_trial.user_attrs.get('rr')}")
        print(f"profit factor: {best_trial.user_attrs.get('pr')}")
        print(f"trades: {best_trial.user_attrs.get('tr')}")
        print(f"Resultado de estabilidad {-best_trial.value}")
        print(f"Mae {best_trial.user_attrs.get('mae')}")
        print(f"Operando {'cortos' if shorts else 'largos'}")

    return study
