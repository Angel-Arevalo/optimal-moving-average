from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Integer, Categorical
import pandas as pd
from read_data import ohlc_form, read_asset
from use_tecnics import main, simple_methods, complex_methods
from tester import backtest_ma
from numpy import exp, log
from typing import Union
import warnings
warnings.filterwarnings("ignore")

weights: list[float] = [0.3, 0.4, 0.3, 0]
calls: int = 50
initial_points: int = 20
lookbacks: int = 110
candles: int = 1
methods: set[str] = complex_methods


def best_main(asset: Union[str, pd.DataFrame], engie: str = "gp", obj: str = "kpi") -> tuple[str, int, int]:
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
                random_state=0,
                verbose=False
        )
    
    
    best_method, best_lookback, best_candle = result.x
    best_obj_value = result.fun
    best_score = -best_obj_value

    return (best_method, best_lookback, best_candle)

def best_partition(asset: Union[str, pd.DataFrame], partitions: int = 3) -> None:
    if isinstance(asset, str):
        data: pd.DataFrame = read_asset(asset)

    else:
        data: pd.DataFrame = asset
    print("filas encontradas del DataFrame", len(data))
    part: int = len(data)//partitions
    pointer: int = part
    for i in range(1, partitions + 1):
        sub_data: pd.DataFrame = data.head(pointer)

        bets_sub: tuple[str, int, int] = best_main(sub_data, "fm", "kpi")
        print("resultados preliminares de entrenamiento", bets_sub)

        ohlc_data: pd.DataFrame = ohlc_form(data, str(bets_sub[2]) + "min")
        ma_perform: pd.Series = main(bets_sub[0], bets_sub[1], ohlc_data)

        kpis: tuple[float, float, float, int] = backtest_ma(ma_perform, ohlc_data["close"], "kpi")

        print("Resultado de entrenamiento con las primeras", pointer, "filas, rango de fechas", 
              f"{sub_data.index[0].strftime("%d/%m/%Y")}-{sub_data.index[-1].strftime("%d/%m/%Y")}")
        print("hit ratio:", kpis[0])
        print("risk reward:", kpis[1])
        print("profit factor:", kpis[2])
        print("trades:", kpis[3], "\n\n")

        pointer += part


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

    if range_candle != 1:
        search_space.append(Integer(1, range_candle, name="candle"))
    else:
        search_space.append(Categorical([1], name="candle"))

    return search_space

def log_prices(asset: Union[str, pd.DataFrame], engie: str = "fm", obj: str = "kpi") -> pd.Series:
    if isinstance(asset, str):
        data: pd.DataFrame =  read_asset(asset)
    else:
        data: pd.DataFrame = asset
    # obtenemos una buena moving averague
    best_method, best_lookback, best_candle = best_main(data, engie, obj)
    
    # Ahora vamos a conseguir el Series que le corresponde a esos 3 
    print(best_candle)
    data = ohlc_form(data, str(best_candle) + "min")
    best_ma: pd.Series = main(best_method, best_lookback, data)
    return log(best_ma).diff()

def realized_variance(asset: Union[str, pd.DataFrame], periods: int = 2):
    # para peridos intradía, es mejor usar periods pequeños,
    # por defecto se usan 2
    
    prices_log: pd.Series = log_prices(asset)
    return (prices_log ** 2).rolling(periods).sum()
