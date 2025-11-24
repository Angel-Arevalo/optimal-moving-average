from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Integer, Categorical
import pandas as pd
from read_data import ohlc_form, read_asset
from use_tecnics import main, simple_methods
from tester import backtest_ma
from numpy import exp
from typing import Union

weights: list[float] = [0.3, 0.4, 0.3, 0]
calls: int = 50
initial_points: int = 20
lookbacks: int = 110
candles: int = 100
methods: set[str] = simple_methods


def best_main(asset: Union[str, pd.DataFrame], engie: str = "gp", obj: str = "kpi") -> None:
    if isinstance(asset, str):
        data: pd.DataFrame = read_asset(asset)
    else:
        data: pd.DataFrame = asset
        
    # función auxiliar para la optimización
    def objective(params: tuple[str, int, int]) -> float:
        method, lookback, vela = params

        return -func_to_opt(data, method, lookback, vela, False, obj)
    
    space: list[Categorical, Integer, Integer] = make_search_space(methods, lookbacks, candles)

    if engie == "gp":
        result = gp_minimize(
                    func=objective,
                    dimensions=space,
                    n_calls=calls,
                    n_initial_points=10,
                    random_state=42,
                )
    if engie == "fm":
        result = forest_minimize(
                func=objective,
                dimensions=space,
                n_calls=calls,
                n_initial_points=10,
                random_state=0
        )
    
    
    best_method, best_lookback, best_candle = result.x
    best_obj_value = result.fun
    best_score = -best_obj_value

    print("Mejor método  :", best_method)
    print("Mejor lookback:", best_lookback)
    print("Mejor candle  :", best_candle)
    print("Best", "KPI combo" if obj == "kpi" else "total money", ":", best_score)
        
    print(func_to_opt(data, best_method, best_lookback, best_candle, True, "kpi"))
    if obj != "mon":
        print(func_to_opt(data, best_method, best_lookback, best_candle, True, "mon"))





def func_to_opt(data: pd.DataFrame, method: str, lookback: int, candle: int, perform: bool = False, obj: str = "kpi") -> float:
    ohlc: pd.DataFrame = ohlc_form(data, str(candle) + "min")
    ma_perform: pd.Series = main(method, lookback, ohlc)
    if obj == "kpi":
        hr, rr, pr, tr = backtest_ma(ma_perform, ohlc["close"])
    
        if perform:
            print("hit ratio, risk reward, profit ratio, trades")
            return (hr, rr, pr, tr)
        return weights[0]*hr + rr*weights[1] + pr*weights[2] + tr*weights[3]
    elif obj == "mon":
        return backtest_ma(ma_perform, ohlc["close"], obj)
    
    return 0

# La cota inferior de range_back es 2 y la de range_candle es 1. La cota máxima depende 
# de los datos usados
def make_search_space(methods: set[str], range_back: int, range_candle: int) -> list[Categorical, Integer, Integer]:
    if methods == None or len(methods) == 0:
        raise ValueError("Lista de métodos no aceptada")

    if range_back <= 1:
        raise ValueError("Invalido espacio de búsqueda para lookback")

    if range_candle <= 0:
        raise ValueError("Inválido espacio de búsqueda para vela")

    search_space: list[Categorical, Integer, Integer] = []
    search_space.append(Categorical(list(methods), name="ma_methods"))
    search_space.append(Integer(2, range_back, name="lookback"))
    search_space.append(Integer(1, range_candle, name="candle"))

    return search_space
