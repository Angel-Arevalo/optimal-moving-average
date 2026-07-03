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


def opti_main(data: Union[pd.DataFrame, str], is_bid: bool = False, verbose: bool = True, engie: str = "fm", shorts: bool = False) -> list:

    if not is_bid and isinstance(data, str):
        data = read_asset(data)

    keys.pre_ohlc = {}
    keys.fill_ohlc_dict(data, is_bid)


    best_result: list = None
    b_met: str = ""
    b_ht: float = 0
    b_rr: float = 0
    b_pr: float = 0
    b_trades: int = 0
    b_score: float = 0
    b_mae: float = 0
    b_mfa: float = 0

    space: list = make_search_space()
    for method in keys.methods:

        def objective(param: list, kpis: bool = True) -> float:
            ohlc: pd.DataFrame = keys.pre_ohlc[param[0]]

            real_param = param[1]

            signals_prices: pd.DataFrame = main(method, ohlc, real_param, shorts, data)

            hr, rr, pr, tr, mae, mfa, sqn = backtest(signals_prices, keys.high_cache[param[0]], keys.low_cache[param[0]], True, shorts)

            if kpis:
                return -f(hr, rr, pr, tr, mae, mfa)

            return (sqn, hr, rr, pr, tr, mae, mfa)

        result: list = optimizer(objective, space, engie)

        if best_result is None:
            best_result = result
            b_score, b_ht, b_rr, b_pr, b_trades, b_mae, b_mfa = objective(result, False)
            b_met = method

        else:
            score, ht, rr, pr, tr, mae, mfa = objective(result, False)

            if b_score < score:
                b_score = score
                b_ht = ht
                b_rr = rr 
                b_pr = pr
                b_trades= tr
                best_result = result
                b_met = method
                b_mae = mae
                b_mfa = mfa

    if verbose:
        #print(f'Resultado obtenido entrenando desde {data.index[0].strftime("%Y-%m-%d")} hasta {data.index[-1].strftime("%Y-%m-%d")}')
        print(f"Método: {b_met}, Datos optimizados {best_result}")
        print(f"\nhit ratio: {b_ht}\nrisk reward: {b_rr}\nprofit factor: {b_pr}\ntrades: {b_trades}")
        print(f"Resultado de estabilidad {b_score}")
        print(f"Mae {b_mae}")
        print(f"Mfa {b_mfa}")
        print(f"Operando {"cortos" if shorts else "largos"}" )


    best_result.insert(0, b_met)

    return best_result

def f(hr: float, rr: float, pr: float, tr: int, mae: float, mfa: float) -> float:


    from numpy import exp, tanh, array, sum as np_sum, clip

    hr_norm = clip(hr_norm, 0.0, 1.0)
    s1 = hr_norm

    pf_clipped = clip(pr, 0.0, 10.0)
    alpha = 2.5
    pf_0 = 1.5
    s2 = 1.0 / (1.0 + exp(-alpha * (pf_clipped - pf_0)))

    rr_clipped = clip(rr, 0.0, 10.0)
    beta = 0.5
    s3 = tanh(beta * rr_clipped)

    s4 = mfa / (mfa + mae + 1e-6)

    gamma_mfe = 0.5
    s5 = 1.0 - exp(-gamma_mfe * mfa)

    lambda_mae = 1.0
    s6 = exp(-lambda_mae * mae)

    sub_scores = array([s1, s2, s3, s4, s5, s6])

    weights = array([0.25, 0.20, 0.20, 0.15, 0.10, 0.10])

    mean_weighted = np_sum(sub_scores * weights)

    variance_weighted = np_sum(weights * (sub_scores - mean_weighted) ** 2)

    delta_penalty = 1.5
    score_final = mean_weighted - (delta_penalty * variance_weighted)

    return float(max(score_final, 0.0))


def optimizer(objective: Callable, space: list, engie: str = "fm") -> tuple:
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

# Ahora se piensa que el espacio depende el método a usarse,
# en el sentido si es método complejo o no

# Se asume que el primer elemento de extras es el elemento que 
# equivale a lookbac de cada método 

def make_search_space() -> list:
    search_space: list = []

    if keys.lookbacks <= 1:
        raise ValueError("Invalido espacio de búsqueda para lookback")

    if keys.candles <= 0:
        raise ValueError("Inválido espacio de búsqueda para vela")

    if keys.candles != 1:
        search_space.append(Integer(1, keys.candles, name="candle"))
    else:
        search_space.append(Categorical([1], name="candle"))

    search_space.append(Integer(2, keys.lookbacks, name="lookback"))

    return search_space
