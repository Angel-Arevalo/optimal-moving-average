from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Integer, Categorical
import pandas as pd
import numpy as np

from read_data import ohlc_form, read_asset
from use_tecnics import main, complex_methods
from tester import backtest

from typing import Union, Callable

import keys
import warnings
warnings.filterwarnings("ignore")


# =========================
# OPTIMIZADOR PRINCIPAL
# =========================
def opti_main(data: Union[pd.DataFrame, str], verbose: bool = True, engie: str = "fm"):

    if isinstance(data, str):
        data = read_asset(data)

    keys.pre_ohlc = {}
    keys.fill_ohlc_dict(data)

    best_result = None
    best_method = ""
    best_score = -np.inf
    best_metrics = None

    for method in keys.methods:

        space = make_search_space(method)

        def objective(param: list) -> float:
            ohlc: pd.DataFrame = keys.pre_ohlc[param[0]]

            if method not in complex_methods:
                real_param = param[1]
            else:
                real_param = param[1:]

            # ajuste MACD si aplica
            if method == "MACD":
                if real_param[0] <= real_param[1]:
                    real_param[0], real_param[1] = real_param[1], real_param[0]

            signals_prices = main(method, ohlc, real_param)

            sharpe, mu, sigma, tr, returns = backtest(signals_prices)

            if tr < 20 or mu <= 0:
                return 10

            penalty = 0.0002 * tr

            score = sharpe - penalty

            return -score

        result = optimizer(objective, space, engie)

        ohlc: pd.DataFrame = keys.pre_ohlc[result[0]]

        if method not in complex_methods:
            real_param = result[1]
        else:
            real_param = result[1:]

        signals_prices = main(method, ohlc, real_param)

        sharpe, mu, sigma, tr, _ = backtest(signals_prices)

        penalty = 0.0002 * tr
        score = sharpe - penalty

        if score > best_score:
            best_score = score
            best_result = result
            best_method = method
            best_metrics = (sharpe, mu, sigma, tr)

    if verbose:
        print("\n=========================")
        print("RESULTADO FINAL")
        print("=========================")
        print(f"Método: {best_method}")
        print(f"Params: {best_result}")
        print(f"Score: {best_score:.4f}")
        print(f"Sharpe: {best_metrics[0]:.4f}")
        print(f"Retorno medio: {best_metrics[1]:.6f}")
        print(f"Volatilidad: {best_metrics[2]:.6f}")
        print(f"Trades: {best_metrics[3]}")

    best_result.insert(0, best_method)
    return best_result

def optimizer(objective: Callable, space: list, engie: str = "fm"):

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