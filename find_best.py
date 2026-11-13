from skopt import gp_minimize, forest_minimize
from skopt.space import Integer, Categorical
from typing import Callable
import numpy as np
import keys
from use_tecnics import generate
from tester import backtest, Metrics


def score(metrics: Metrics) -> float:
    return f(**metrics._asdict())


def opti_main(data, verbose: bool = True, engie: str = "gp", shorts: bool = False, asset: str = None) -> list:
    df = keys.fill_ohlc_dict(data, asset)
    space = make_search_space()

    def evaluate(params: list) -> Metrics:
        method, candle, lookback = params
        return backtest(generate(method, lookback, candle, shorts), df, shorts)

    result = optimizer(lambda params: -score(evaluate(params)), space, engie)
    if verbose:
        print(f"Método y parámetros: {result}")
        print(evaluate(result)._asdict())
        print(f"Operando {'cortos' if shorts else 'largos'}")
    return result


def softplus(x: float, k: float = 1.0) -> float:
    kx = k * x
    if kx > 30.0:
        return float(x)
    elif kx < -30.0:
        return 0.0
    return float(np.log1p(np.exp(kx)) / k)

def f(
    hr: float, 
    rr: float, 
    pr: float, tr: int, 
    sqn: float, mae: float = 0.0) -> float:
    hr_safe = np.clip(hr, 0.01, 0.99)
    rr_safe = max(rr, 0.01)
    pr_safe = max(pr, 0.01)

    e_stab = (hr_safe * np.log1p(rr_safe)) - (1.0 - hr_safe)
    g_e = softplus(e_stab, k=10.0)

    p_balance = np.sqrt(hr_safe / (1.0 - hr_safe))

    s_pr = np.log1p(softplus(pr_safe - 1.0, k=5.0))

    c_tr = 10.0 * np.tanh(tr / 30.0) * (1.0 - np.exp(-tr / 10.0))

    sqn_positive = softplus(sqn, k=1.0)
    s_sqn = 5.0 * np.tanh(sqn_positive / 5.0)

    mae_delta = softplus(mae - 0.50, k=4.0)
    eta_mae = np.exp(-0.15 * mae_delta)

    fitness_score = g_e * p_balance * s_pr * c_tr * (1.0 + 0.25 * s_sqn) * eta_mae

    return float(fitness_score)

def optimizer(objective: Callable, space: list, engie: str = "gp") -> list:
    engines = {"gp": gp_minimize, "fm": forest_minimize}
    if engie not in engines:
        raise ValueError(f"No se reconoce {engie} como motor")
    if keys.calls < 1 or keys.initial_points < 1:
        raise ValueError("calls e initial_points deben ser positivos")
    options = {"n_calls": keys.calls, "n_initial_points": min(keys.initial_points, keys.calls), "random_state": 42}
    return engines[engie](objective, space, **options).x


def make_search_space() -> list:
    keys.validate_space()
    def dimension(low, high, name):
        return Categorical([low], name=name) if low == high else Integer(low, high, name=name)
    return [Categorical(sorted(keys.methods), name="name"),
            dimension(keys.candles_min, keys.candles, "candle"),
            dimension(keys.lookbacks_min, keys.lookbacks, "lookback")]
