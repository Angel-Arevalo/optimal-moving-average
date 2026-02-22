from skopt import gp_minimize, forest_minimize
from skopt.space import Integer
import pandas as pd
from warnings import filterwarnings
from rsi_atr_opt.sub_tester import test_ma_rsi
filterwarnings("ignore")

max_n: int = 50
calls: int = 50

# se espera que el primer argumento debe ser una moving 
# averague, 
def main(data: pd.DataFrame, objetive: str, moving_averague: pd.Series = None, verbose: bool = False, engie: str = "fm"):
    def f(n):
        if objetive == "rsi":
            return -func_to_opt_rsi(data["close"], moving_averague, n[0])
        else:
            raise ValueError("Solo es posible optimizar rsi o atr")

    space: list[Integer] = make_search_space(max_n)

    if engie == "fm":
        result = forest_minimize(
                 func = f,
                 dimensions = space,
                 n_calls = calls,
                 n_initial_points = 10,
                 random_state = 0,
                 verbose = False
        )
    elif engie == "gp":
        result = gp_minimize(
            func = f,
            dimensions = space,
            n_calls = calls,
            n_initial_points = 10,
            random_state = 0,
            verbose = False
        )
    else:
        raise ValueError("Motor de búsqueda no conocido")

    best_n: int = result.x[0]
    return best_n
    

def make_search_space(range_n: int) -> list[Integer]:
    if range_n <= 2:
        raise ValueError("Rango pequeño de busqueda")

    search_space: list[Integer] = []
    search_space.append(Integer(2, range_n, name = "valor de n"))

    return search_space

def func_to_opt_rsi(real_data: pd.Series, ma: pd.Series, n: int) -> float:
    hr, rr, pf, t = test_ma_rsi(real_data, ma, n)

    return .4*hr + .3*rr + .3*pf
