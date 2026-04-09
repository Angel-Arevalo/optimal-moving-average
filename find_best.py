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


def opti_main(data: Union[pd.DataFrame, str], is_bid: bool = False, verbose: bool = True, engie: str = "fm") -> list:
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

    for method in keys.methods:
        space: list = make_search_space(method)

        def objective(param: list, kpis: bool = True) -> float:
            ohlc: pd.DataFrame = keys.pre_ohlc[param[0]]

            real_param = param[1]

            signals_prices: pd.DataFrame = main(method, ohlc, real_param)

            hr, rr, pr, tr, sqn = backtest(signals_prices, True)
            if kpis:
                return -f(hr, rr, pr, tr, sqn)

            return (sqn, hr, rr, pr, tr)

        result: list = optimizer(objective, space, engie)

        if best_result is None:
            best_result = result
            b_score, b_ht, b_rr, b_pr, b_trades = objective(result, False)
            b_met = method

        else:
            score, ht, rr, pr, tr = objective(result, False)

            if b_score < score:
                b_score = score
                b_ht = ht
                b_rr = rr 
                b_pr = pr
                b_trades= tr
                best_result = result
                b_met = method

    if verbose:
        #print(f'Resultado obtenido entrenando desde {data.index[0].strftime("%Y-%m-%d")} hasta {data.index[-1].strftime("%Y-%m-%d")}')
        print(f"Método: {b_met}, Datos optimizados {best_result}")
        print(f"\nhit ratio: {b_ht}\nrisk reward: {b_rr}\nprofit factor: {b_pr}\ntrades: {b_trades}")
        print(f"Resultado de sobre ajuste {b_score}")

    best_result.insert(0, b_met)

    return best_result

def f(hr: float, rr: float, pr: float, tr: int, sqn: float) -> float:
    if rr < 1.0 or pr < 1.1:
        return 10

    loss = 1 - hr

    expecty = hr*rr - loss

    expecty = expecty*sqrt(tr)

    if pr > 0:
        expecty += log(pr)

    return expecty

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

def make_search_space(method: str) -> list:
    search_space: list = []

    if method not in keys.methods:
        raise ValueError("Método no aceptado")

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

# Cada resultado tiene la forma [metodo, candle, añadidos]
def read_results(result: list, real_data: pd.DataFrame) -> None:
    ohlc = ohlc_form(real_data, str(result[1]) + "min")["close"]

    vector_perform: pd.DataFrame = main(result[0], ohlc, result[2:])

    hr, rr, pr, tr = backtest(vector_perform)

    print("Rendimiento de una ma con:")
    print(f"método: {result[0]}, vela: {result[1]}, añadidos: {result[2:]}")
    print(f"hit ratio: {hr}\nrisk reward: {rr}\nprofit factor: {pr}\ntrades: {tr}\n\n")
