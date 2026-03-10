from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Integer, Categorical
import pandas as pd

from read_data import ohlc_form, read_asset
from use_tecnics import main, complex_methods, avalible_methods
from tester import backtest

from numpy import exp, log, sqrt
from typing import Union, Callable

import keys
import warnings
warnings.filterwarnings("ignore")


def opti_main(data: Union[pd.DataFrame, str], engie: str = "fm") -> list:
    if isinstance(data, str):
        data = read_asset(data)

    keys.pre_ohlc = {}
    keys.fill_ohlc_dict(data)

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

            if method not in complex_methods:
                real_param = param[1]
            else:
                real_param: list = param[1:]

            if method == "MACD":
                if real_param[0] <= real_param[1]:
                    temp: int = real_param[0]
                    real_param[0] = real_param[1]
                    real_param[1] = temp

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

    print(f'Resultado obtenido entrenando desde {data.index[0].strftime("%Y-%m-%d")} hasta {data.index[-1].strftime("%Y-%m-%d")}')
    print(f"Método: {b_met}, Datos optimizados {best_result}")
    print(f"\nhit ratio: {b_ht}\nrisk reward: {b_rr}\nprofit factor: {b_pr}\ntrades: {b_trades}")
    print(f"Resultado de sobre ajuste {score}")
    best_result.insert(0, b_met)

    return best_result

def f(hr: float, rr: float, pr: float, tr: int, sqn: float) -> float:
    if pr < 1.0 or sqn < 0:
        return 1

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
                    random_state=0,
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

    if method in complex_methods:
        if method == "MACD":
            search_space.append(Integer(2, keys.lookbacks_min, name="lookback_min"))
            search_space.append(Integer(2, keys.signal_back, name="signal_back"))

        elif method == "BBANDS":
            search_space.append(Real(1.0, keys.dev_up, name="dev_up"))
            search_space.append(Real(1.0, keys.dev_dn, name="dev_dn"))
            search_space.append(Categorical(list(range(keys.matype + 1)), name="matype"))

        elif method == "DONCHIAN":
            search_space.append(Integer(2, keys.lookbacks_min, name="lookback_lower"))

        elif method == "ZSCORE_EMA":
            search_space.append(Real(0.0, keys.threshold, name="threshold"))
            search_space.append(Categorical(list(range(keys.matype + 1)), name="matype"))

        else:
            raise ValueError(f"{method} no es un método implementado aún")

    return search_space

# Cada resultado tiene la forma [metodo, candle, añadidos]
def read_results(result: list, real_data: pd.DataFrame) -> None:
    ohlc = ohlc_form(real_data, str(result[1]) + "min")["close"]

    vector_perform: pd.DataFrame = main(result[0], ohlc, result[2:])

    hr, rr, pr, tr = backtest(vector_perform)

    print("Rendimiento de una ma con:")
    print(f"método: {result[0]}, vela: {result[1]}, añadidos: {result[2:]}")
    print(f"hit ratio: {hr}\nrisk reward: {rr}\nprofit factor: {pr}\ntrades: {tr}\n\n")
