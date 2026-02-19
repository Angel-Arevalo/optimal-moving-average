from skopt import gp_minimize, forest_minimize
from skopt.space import Integer
import pandas as pd
from warnings import filterwarnings
filterwarnings("ignore")

max_n: int = 50
calls: int = 50

# se espera que el atr y rsi 
# tengan uso en la búsqueda en algún futuro, 
# el valor puede cambiarse según el modelo
horizon: int = 5

# se espera que el primer argumento debe ser una moving 
# averague, 
def main(data: pd.DataFrame, objetive: str, moving_averague: pd.Series = None, engie: str = "fm"):
    def f(n: int):
        if objetive == "rsi":
            return -func_to_opt_rsi(data["close"], moving_averague, n[0])
        elif objetive == "atr":
            return -func_to_opt_atr(data["close"], data, n[0])
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

def compute_rsi(series: pd.Series, n: int) -> pd.Series:
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/n, adjust=False).mean()

    rs = avg_gain.div(avg_loss.replace(0, 1e-10))
    rsi = 100 - (100 / (1 + rs))

    return rsi

def func_to_opt_rsi(price: pd.Series, ma: pd.Series, n: int) -> float:
    rsi = compute_rsi(ma, n)

    future_returns = price.pct_change().shift(-horizon)

    data = pd.concat([rsi, future_returns], axis=1)
    data.columns = ["rsi", "future_returns"]
    data = data.dropna()

    ic = data["rsi"].corr(data["future_returns"])

    if pd.isna(ic):
        return 0.0

    return abs(ic)

def compute_atr(df: pd.DataFrame, n: int) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.ewm(alpha=1/n, adjust=False).mean()
    
    return atr

def func_to_opt_atr(price: pd.Series, df: pd.DataFrame, n: int) -> float:
    atr = compute_atr(df, n)

    future_returns = price.pct_change().shift(-horizon)

    data = pd.concat([atr, future_returns], axis=1)
    data.columns = ["atr", "future_returns"]
    data = data.dropna()

    ic = data["atr"].corr(data["future_returns"])

    if pd.isna(ic):
        return 0.0

    return abs(ic)
