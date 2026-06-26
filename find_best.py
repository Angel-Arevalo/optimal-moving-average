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

    space: list = make_search_space()
    for method in keys.methods:

        def objective(param: list, kpis: bool = True) -> float:
            ohlc: pd.DataFrame = keys.pre_ohlc[param[0]]

            real_param = param[1]

            signals_prices: pd.DataFrame = main(method, ohlc, real_param, shorts, data)

            if shorts:
                hr, rr, pr, tr, mae, sqn = backtest(signals_prices, keys.high_cache[param[0]], True, shorts)
            else:
                 hr, rr, pr, tr, mae, sqn = backtest(signals_prices, keys.low_cache[param[0]], True, shorts)

            if kpis:
                return -f(hr, rr, pr, tr, mae)

            return (sqn, hr, rr, pr, tr, mae)

        result: list = optimizer(objective, space, engie)

        if best_result is None:
            best_result = result
            b_score, b_ht, b_rr, b_pr, b_trades, b_mae = objective(result, False)
            b_met = method

        else:
            score, ht, rr, pr, tr, mae = objective(result, False)

            if b_score < score:
                b_score = score
                b_ht = ht
                b_rr = rr 
                b_pr = pr
                b_trades= tr
                best_result = result
                b_met = method
                b_mae = mae

    if verbose:
        #print(f'Resultado obtenido entrenando desde {data.index[0].strftime("%Y-%m-%d")} hasta {data.index[-1].strftime("%Y-%m-%d")}')
        print(f"Método: {b_met}, Datos optimizados {best_result}")
        print(f"\nhit ratio: {b_ht}\nrisk reward: {b_rr}\nprofit factor: {b_pr}\ntrades: {b_trades}")
        print(f"Resultado de estabilidad {b_score}")
        print(f"Mae {b_mae}")
        print(f"Operando {"cortos" if shorts else "largos"}" )

    best_result.insert(0, b_met)

    return best_result

def f(hr: float, rr: float, pr: float, tr: int, mae: float) -> float:
    expectancy = hr * rr - (1 - hr)

    if expectancy <= 0 or pr <= 1.0:
        return -1000

    kelly = expectancy / rr

    confidence = sqrt(min(tr, 100))

    pf_bonus = log(pr)

    efficiency = expectancy / (expectancy + mae)

    return kelly * confidence * pf_bonus * efficiency

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
