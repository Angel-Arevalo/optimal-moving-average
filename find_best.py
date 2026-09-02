from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Integer, Categorical
import pandas as pd

from read_data import ohlc_form, read_asset
from use_tecnics import main, avalible_methods
from tester import backtest

from numpy import exp, log, sqrt
from typing import Union, Callable
import numpy as np

import keys
import warnings
warnings.filterwarnings("ignore")


def opti_main(data: Union[pd.DataFrame, str], verbose: bool = True, engie: str = "gp", shorts: bool = False) -> list:

    keys.pre_ohlc = {}
    keys.fill_ohlc_dict(data)

    space: list = make_search_space()

    def objective(param: list, kpis: bool = True) -> float:
        method = param[0]

        ohlc: pd.DataFrame = keys.mid_cache[param[1]]["close"]

        signals_prices: pd.DataFrame = main(method, ohlc, param[2], param[1], data, shorts)

        if shorts:
            hr, rr, pr, tr, mae, sqn = backtest(signals_prices, keys.bid_cache[param[1]]["high"], True, shorts)
        else:
            hr, rr, pr, tr, mae, sqn = backtest(signals_prices, keys.ask_cache[param[1]]["low"], True, shorts)

        if kpis:
            return -f(hr, rr, pr, tr, sqn, mae)

        return (sqn, hr, rr, pr, tr, mae)

    result: list = optimizer(objective, space, engie)

    best_result = result
    b_score, b_ht, b_rr, b_pr, b_trades, b_mae = objective(result, False)
    b_met = result[0]

    if verbose:
        #print(f'Resultado obtenido entrenando desde {data.index[0].strftime("%Y-%m-%d")} hasta {data.index[-1].strftime("%Y-%m-%d")}')
        print(f"Método: {b_met}, Datos optimizados {best_result}")
        print(f"\nhit ratio: {b_ht}\nrisk reward: {b_rr}\nprofit factor: {b_pr}\ntrades: {b_trades}")
        print(f"Resultado de estabilidad {b_score}")
        print(f"Mae {b_mae}")
        print(f"Operando {"cortos" if shorts else "largos"}" )

    return best_result

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

    e_stab = (hr_safe * rr_safe) - (1.0 - hr_safe)
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

def optimizer(objective: Callable, space: list, engie: str = "gp") -> tuple:
    if engie == "gp":
        result = gp_minimize(
                    func=objective,
                    dimensions=space,
                    n_calls=keys.calls,
                    n_initial_points=10,
                    random_state=42,
                )

    elif engie == "fm":
        result = forest_minimize(
                    func=objective,
                    dimensions=space,
                    n_calls=keys.calls,
                    n_initial_points=10,
                    random_state=None,
                    verbose=False
                 )

    else:
        raise ValueError(f"No se reconoce {engie} como Motor")

    return result.x


def make_search_space() -> list:
    search_space: list = []

    search_space.append(Categorical(list(keys.methods), name="name"))

    if keys.lookbacks_min < 2:
        raise ValueError("Espacio pequeño de vista hacia atras")

    if keys.lookbacks < keys.lookbacks_min:
        raise ValueError("Espacio de búsqueda inválido para lookback (min > max)")

    if keys.candles < keys.candles_min:
        raise ValueError("Espacio de búsqueda inválido para vela (min > max)")

    if keys.candles_min == keys.candles:
        search_space.append(Categorical([keys.candles_min], name="candle"))
    else:
        search_space.append(Integer(keys.candles_min, keys.candles, name="candle"))

    if keys.lookbacks_min == keys.lookbacks:
        search_space.append(Categorical([keys.lookbacks_min], name="lookback"))
    else:
        search_space.append(Integer(keys.lookbacks_min, keys.lookbacks, name="lookback"))

    return search_space
