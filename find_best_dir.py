from typing import List, Tuple, Dict, Any
import copy
import numpy as np
import optuna
import pandas as pd

from direction_methods import dir_main
import keys

from tester import backtest
from find_best import DEFAULT_SEED, ordered_ma_methods, softplus, f as f_single
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
    "OU_REVERSION",
    "LINREG_R2",
    "LINREG_SLOPE",
    "HT_TRENDMODE",
    "AROONOSC",
    "DI_BALANCE",
    "DRIFT_PROB",
]

optuna.logging.set_verbosity(optuna.logging.WARNING)

def opti_dir(asset, verbose: bool = True, shorts: bool = False, n_trials: int = 100,
             *, seed: int = DEFAULT_SEED, filter_side: str = "both", weekly_closes: bool = True) -> Tuple:
    if filter_side not in ("both", "revert", "tend"):
        raise ValueError("filter_side must be 'both', 'revert' or 'tend'")
    keys.bid_cache = {}
    keys.ask_cache = {}
    keys.mid_cache = {}

    daily = keys.fill_ohlc_dict(asset)

    ma_candles = [i for i in keys.mid_cache if i >= keys.candles_min]
    dir_candles = [i for i in keys.mid_cache if i <= 100] or ma_candles
    ma_methods = ordered_ma_methods()

    # Los detectores de volumen sólo entran si el cache trae volumen.
    from direction_methods import VOLUME_DIR_METHODS
    dir_methods_eff = list(dir_methods)
    if keys.has_volume:
        dir_methods_eff += sorted(VOLUME_DIR_METHODS)

    completed_scores: List[float] = []
    score_cache: Dict[tuple, tuple] = {}

    def suggest_candle(trial, name, values):
        if len(values) == values[-1] - values[0] + 1:
            return trial.suggest_int(name, values[0], values[-1])
        return trial.suggest_categorical(name, values)

    def objective(trial: optuna.Trial) -> float:
        ma_method = trial.suggest_categorical(
            "ma_method", ma_methods
        )
        ma_candle = suggest_candle(trial, "ma_candle", ma_candles)
        ma_lookback = trial.suggest_int(
            "ma_lookback", keys.lookbacks_min, keys.lookbacks
        )

        ohlc: pd.DataFrame = keys.mid_cache[ma_candle]["close"]
        signals_prices: pd.DataFrame = main(ma_method, ohlc, ma_lookback, ma_candle, asset, shorts, daily=daily, weekly_closes=weekly_closes)

        if shorts:
            initial_params: tuple = backtest(signals_prices, keys.bid_cache[ma_candle]["high"], True, shorts)
        else:
            initial_params: tuple = backtest(signals_prices, keys.ask_cache[ma_candle]["low"], True, shorts)

        dir_method = trial.suggest_categorical("name", dir_methods_eff)
        dir_candle = suggest_candle(trial, "candle", dir_candles)
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
        elif dir_method == "OBV_SLOPE":
            params["follow_tend"] = trial.suggest_float("follow_tend", 0.0, 1.0)
        elif dir_method == "MFI":
            # Distancia al centro 50 (0 a 50).
            params["follow_tend"] = trial.suggest_float("follow_tend", 0.0, 40.0)
        elif dir_method == "ADOSC":
            params["fastperiod"] = trial.suggest_int("fastperiod", 2, 20)
            params["slowperiod"] = trial.suggest_int("slowperiod", 10, 60)
            params["follow_tend"] = trial.suggest_float("follow_tend", 0.0, 1.0)
        elif dir_method == "VWAP_DIST":
            params["follow_tend"] = trial.suggest_float("follow_tend", 0.0, 0.1)
        elif dir_method == "MARKETFI":
            params["follow_tend"] = trial.suggest_float("follow_tend", 0.5, 3.0)
        elif dir_method == "VOL_SURGE":
            params["follow_tend"] = trial.suggest_float("follow_tend", 1.0, 4.0)
        elif dir_method == "LINREG_R2":
            # R² entre 0 y 1.
            params["follow_tend"] = trial.suggest_float("follow_tend", 0.1, 0.95)
        elif dir_method == "LINREG_SLOPE":
            # Pendiente normalizada por precio (fracción por barra).
            params["follow_tend"] = trial.suggest_float("follow_tend", 0.0, 0.05)
        elif dir_method == "HT_TRENDMODE":
            # Clasificador binario nativo: sin umbral ni ventana propios.
            pass
        elif dir_method == "AROONOSC":
            # Oscilador -100 a 100; umbral sobre su magnitud.
            params["follow_tend"] = trial.suggest_float("follow_tend", 0.0, 80.0)
        elif dir_method == "DI_BALANCE":
            # |+DI - -DI|, escala 0 a 100.
            params["follow_tend"] = trial.suggest_float("follow_tend", 0.0, 50.0)
        elif dir_method == "DRIFT_PROB":
            # Horizonte de proyección del camino y umbral de sesgo |p-0.5| (0 a 0.5).
            params["horizon"] = trial.suggest_int("horizon", 1, 20)
            params["follow_tend"] = trial.suggest_float("follow_tend", 0.0, 0.45)

        filter_params: tuple = dir_main(signals_prices, asset, ma_candle, params, shorts, filter_side, daily=daily)

        hr_dir, rr_dir, pr_dir, tr_dir, mae_dir, sqn_dir, intervenidos, _, _ = filter_params

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
        trial.set_user_attr("tr_int", intervenidos)

        return -f(initial_params, filter_params, filter_side=filter_side)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
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
        "tr_inter": best_trial.user_attrs.get("tr_int")
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
        tr_i  = best_metrics.get('tr_inter') if best_metrics.get('tr_inter') is not None else "N/A"
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
        tr_f_str = f"{tr_i} mod / {tr_f}" 

        print(f"{'Trades':<15} | {str(tr_0):>15} | {tr_f_str:>15}")
        print(f"{'SQN':<15} | {sqn_0:>15.4f} | {sqn_f:>15.4f}")
        print(f"{'MAE':<15} | {mae_0:>15.5f} | {mae_f:>15.5f}")
        print("-" * 49 + "\n")

    return best_params, best_metrics

def f(initial_params: Tuple[float, ...], filter_params: Tuple[float, ...], *, filter_side="both") -> float:
    hr_0, rr_0, pr_0, tr_0, mae_0, sqn_0 = initial_params

    hr_f, rr_f, pr_f, tr_f, mae_f, sqn_f, intervened, f_1, f_2 = filter_params
    k = 5

    # El detector debe separar bien AMBOS lados (reversión f_1 y tendencia f_2),
    # sin importar cuál se descarte luego: un modelo sin sesgo es bueno en los dos.
    # El soft-min premia que el peor de los dos también sea alto; si el lado que
    # se descartaría era malo, el score baja igual que si fallara el que se conserva.
    comb_fitness = -1 / k * np.log(np.exp(-k * f_1) + np.exp(-k * f_2))
    eps = 1e-6

    q_base = f_single(hr_0, rr_0, pr_0, int(tr_0), sqn_0, mae_0)
    q_filtered = f_single(hr_f, rr_f, pr_f, int(tr_f), sqn_f, mae_f)

    tr_0_safe = max(tr_0, 1.0)
    intervention_ratio = intervened / tr_0_safe

    gate_lower = 1.0 / (1.0 + np.exp(-40.0 * (intervention_ratio - 0.05)))

    gate_upper = 1.0 / (1.0 + np.exp(40.0 * (intervention_ratio - 0.95)))

    activity_gate = gate_lower * gate_upper

    delta_pr = pr_f - pr_0
    differentiation_reward = np.log1p(softplus(delta_pr, k=2.0))

    r_improvement = (q_filtered + eps) / (q_base + eps)

    d_hr = hr_f - hr_0
    gate_hr = 1.0 / (1.0 + np.exp(-100.0 * d_hr))

    # Multiplicador de confianza (sigmoide): se fía de los KPIs solo si la
    # estrategia final alcanza el estándar de actividad. Estándar: ~1 trade cada
    # 2 semanas = 28/año; con 3 años de train, 28*3 = 84 operaciones es el centro.
    # Por debajo de 84 el score se descuenta fuerte; por encima satura a ~1 sin
    # comprimir a los candidatos con muchas operaciones (a diferencia de un tanh
    # escalado, que achataba las diferencias por arriba).
    # sigmoide(tr_f): 84 -> 0.50, 120 -> 0.86, 40 -> 0.10, 12 -> 0.03, 270 -> ~1.
    confidence_center = 84.0
    confidence_slope = 0.05
    confidence = float(1.0 / (1.0 + np.exp(-confidence_slope * (tr_f - confidence_center))))

    final_score = (q_filtered * r_improvement * gate_hr * activity_gate
                   * (1.0 + differentiation_reward) * comb_fitness * confidence)

    return float(final_score)
