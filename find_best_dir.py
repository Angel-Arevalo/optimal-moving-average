from numbers import Integral
import numpy as np
import optuna
import keys
from direction_methods import dir_main, DIR_METHODS
from find_best import score
from use_tecnics import generate
from tester import backtest


dir_methods = list(DIR_METHODS)
THRESHOLDS = {"KEF": (.1, 1.), "HURST": (.5, 1.), "LO_MACKINLAY": (.5, 3.),
              "ADX": (10., 40.), "VOL_RATIO": (.5, 3.), "SHANNON": (.3, 1.), "ATR_EXPANSION": (1., 3.)}


def improvement_objective(base_fitness: float, final_fitness: float, improvement_weight: float = 0.75) -> tuple[float, float]:
    """Combine final quality with signed normalized gain, and return that gain.

    The gain lies in [-1, 1]. It is not a percentage return or a significance test.
    """
    if not np.isfinite(improvement_weight) or not 0 <= improvement_weight <= 1:
        raise ValueError("improvement_weight debe estar entre 0 y 1")
    if not np.isfinite([base_fitness, final_fitness]).all() or min(base_fitness, final_fitness) < 0:
        raise ValueError("Los fitness deben ser finitos y no negativos")
    scale = max(base_fitness, final_fitness)
    if scale == 0:
        return 0.0, 0.0
    base, final = base_fitness / scale, final_fitness / scale
    gain = (final - base) / (final + base)
    factor = (1.0 - improvement_weight) + improvement_weight * gain
    return float(final_fitness * factor), float(gain)


def make_cache_key(method: str, candle: int, lookback: int, params: dict, filter_val: bool) -> tuple:
    return method, candle, lookback, filter_val, tuple(sorted(params.items()))


def generate_deterministic_neighbors(params: dict) -> list[dict]:
    bounds = {"candle": (keys.candles_min, keys.candles, int), "window": (2, 100, int),
              "follow_tend": (0.01, 50.0, float), "k": (2, 10, int), "bins": (5, 20, int),
              "lookback_ma": (20, 100, int), "mult": (1.0, 4.0, float)}
    if params["name"] in THRESHOLDS:
        bounds["follow_tend"] = (*THRESHOLDS[params["name"]], float)
    neighbors = [params.copy()]
    for step, delta in [(1, .05), (-1, -.05), (2, .10), (-2, -.10), (1, -.05), (-1, .05), (3, .15), (-3, -.15)]:
        config = params.copy()
        for key, value in params.items():
            if key in bounds:
                low, high, dtype = bounds[key]
                value = value + step if dtype is int else value * (1 + delta)
                config[key] = dtype(np.clip(value, low, high))
        neighbors.append(config)
    return neighbors


def suggest_params(trial: optuna.Trial) -> dict:
    name = trial.suggest_categorical("name", dir_methods)
    params = {"name": name, "candle": trial.suggest_int("candle", keys.candles_min, keys.candles),
              "window": trial.suggest_int("window", 2, 100)}
    thresholds = THRESHOLDS
    extras = {"LO_MACKINLAY": ("k", 2, 10), "SHANNON": ("bins", 5, 20), "ATR_EXPANSION": ("lookback_ma", 20, 100)}
    if name in thresholds:
        params["follow_tend"] = trial.suggest_float("follow_tend", *thresholds[name])
    if name in extras:
        key, low, high = extras[name]
        params[key] = trial.suggest_int(key, low, high)
    if name == "KELTNER_BREAKOUT":
        params["mult"] = trial.suggest_float("mult", 1., 4.)
    return params


def opti_dir(data, verbose: bool = True, shorts: bool = False, filter: bool = False, n_trials: int = 100, asset: str = None, ma_seed: list = None, improvement_weight: float = 0.75) -> tuple:
    if isinstance(n_trials, bool) or not isinstance(n_trials, Integral) or n_trials < 1:
        raise ValueError("n_trials debe ser un entero positivo")
    if filter is not None and not isinstance(filter, bool):
        raise ValueError("filter debe ser True, False o None")
    improvement_objective(0., 0., improvement_weight)
    df = keys.fill_ohlc_dict(data, asset)
    cache = {}
    base_cache = {}

    def objective(trial: optuna.Trial) -> float:
        method = trial.suggest_categorical("ma_method", sorted(keys.methods))
        candle = trial.suggest_int("ma_candle", keys.candles_min, keys.candles)
        lookback = trial.suggest_int("ma_lookback", keys.lookbacks_min, keys.lookbacks)
        ma_key = method, candle, lookback
        if ma_key not in base_cache:
            signals = generate(method, lookback, candle, shorts)
            base_metrics = backtest(signals, df, shorts)
            base_cache[ma_key] = signals, base_metrics, score(base_metrics)
        signals, base_metrics, base_fitness = base_cache[ma_key]
        params = suggest_params(trial)
        filtered = trial.suggest_categorical("filter", [False, True]) if filter is None else filter
        scores = []
        for i, config in enumerate(generate_deterministic_neighbors(params)):
            key = make_cache_key(method, candle, lookback, config, filtered)
            if key not in cache:
                metrics = dir_main(signals, df, config, shorts, filtered)
                cache[key] = score(metrics), metrics
            value, metrics = cache[key]
            combined, gain = improvement_objective(base_fitness, value, improvement_weight)
            scores.append(combined)
            if i == 0:
                trial.set_user_attr("improvement_weight", improvement_weight)
                trial.set_user_attr("base_fitness", base_fitness)
                trial.set_user_attr("final_fitness", value)
                trial.set_user_attr("fitness_gain", value - base_fitness)
                trial.set_user_attr("improvement_factor", gain)
                trial.set_user_attr("combined_fitness", combined)
                trial.set_user_attr("improved", value > base_fitness)
                for name, metric in base_metrics._asdict().items():
                    trial.set_user_attr("base_" + name, metric)
                for name, metric in metrics._asdict().items():
                    trial.set_user_attr(name, metric)
        std = float(np.std(scores))
        trial.set_user_attr("std_score", std)
        return -(float(np.mean(scores)) - 1.5 * std)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    if ma_seed is not None:
        method, candle, lookback = ma_seed
        for name in dir_methods:
            study.enqueue_trial({"ma_method": method, "ma_candle": candle, "ma_lookback": lookback, "name": name})
    study.optimize(objective, n_trials=n_trials)
    best = study.best_trial
    metrics = {"score": -best.value, **best.user_attrs}
    if verbose:
        print(f"Método y parámetros: {best.params}")
        print(metrics)
        print(f"Operando {'cortos' if shorts else 'largos'}")
    return best.params, metrics
