from skopt import gp_minimize, forest_minimize
from skopt.space import Integer
import pandas as pd
from warnings import filterwarnings
filterwarnings("ignore")

max_n: int = 50
calls: int = 50

# se espera que el primer argumento debe ser una moving 
# averague, 
def main(data: pd.Series, moving_averague: pd.Series, engie: str = "fm"):
    def f(n: int):
        return -func_to_opt(data, moving_averague, n[0])

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
    if engie == "gp":
        result = gp_minimize(
            func = f,
            dimensions = space,
            n_calls = calls,
            n_initial_points = 10,
            random_state = 0,
            verbose = False
        )
    
    best_n: int = result.x[0]
    return best_n
    

def make_search_space(range_n: int) -> list[Integer]:
    if range_n <= 2:
        raise ValueError("Rango pequeño de busqueda")

    search_space: list[Integer] = []
    search_space.append(Integer(2, range_n, name = "valor de n"))

    return search_space

def compute_rsi(series: pd.Series, n: int) -> pd.Series:
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/n, adjust=False).mean()

    rs = avg_gain.div(avg_loss.replace(0, 1e-10))
    rsi = 100 - (100 / (1 + rs))

    return rsi

def func_to_opt(price: pd.Series, ma: pd.Series, n: int, k: int = 5) -> float:
    rsi = compute_rsi(ma, n)

    future_returns = price.pct_change().shift(-k)

    df = pd.concat([rsi, future_returns], axis=1).dropna()

    ic = df.iloc[:, 0].corr(df.iloc[:, 1])

    return abs(ic)

