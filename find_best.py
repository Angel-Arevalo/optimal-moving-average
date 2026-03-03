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


def opti_main(data: Union[pd.DataFrame, str], engie: str = "fm",max_rsi: int = None) -> list:
    if isinstance(data, str):
        data = read_asset(data)
    
    best_results: list = []

    for method in keys.methods:
        space: list = make_search_space(method, max_rsi)

        def objective(param: list) -> float:
            ohlc: pd.DataFrame = ohlc_form(data, str(param[0]) + "min")

            if method not in complex_methods:
                real_param = param[1]
            else:
                real_param: list = param[1:] if max_rsi is None else param[1:-1]

            if method == "MACD":
                if real_param[0] <= real_param[1]:
                    temp: int = real_param[0]
                    real_param[0] = real_param[1]
                    real_param[1] = temp

            signals_prices: pd.DataFrame = main(method, ohlc, real_param)

            if max_rsi is not None:
               hr, rr, pr, tr = backtest(signals_prices, param[-1], ohlc["close"])
            else:
                hr, rr, pr, tr = backtest(signals_prices)

            if tr == 0:
                return 1.

            loss = 1 - hr
            expecty = (hr * rr) - loss
            score = expecty * sqrt(tr)

            if pr > 0:
                score += log(pr)

            return -score

        best_results.append({method: optimizer(objective, space, engie)})

    return best_results


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

def make_search_space(method: str, range_rsi: int = None) -> list:
    search_space: list = []

    if method not in keys.methods:
        raise ValueError("Método no aceptado")

    if keys.lookbacks <= 1:
        raise ValueError("Invalido espacio de búsqueda para lookback")

    if keys.candles <= 0:
        raise ValueError("Inválido espacio de búsqueda para vela")

    if range_rsi != None and range_rsi <= 1:
        raise ValueError("Inválido espacio de búsqueda para parámetro de RSI")
    
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

    if range_rsi != None:
        if range_rsi == 2:
            search_space.append(Categorical([2], name="rsi_v"))

        else:
            search_space.append(Integer(2, range_rsi, name="rsi_v"))

    return search_space
