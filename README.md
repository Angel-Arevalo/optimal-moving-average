# Optimización de medias móviles

Backtest por operación para comparar medias móviles y filtros mediante métricas. La búsqueda selecciona método, período y temporalidad; opcionalmente también indicador de dirección y filtrado. No simula una cuenta ni dimensiona posiciones.

## Instalación y uso

```bash
source opt/bin/activate
pip install -r requirements.txt
python test.py precios.csv --calls 60 --temporality min --sessions 60
python test.py precios.csv --direction --calls 100 --temporality min --sessions 60
```

`--sessions` fija el máximo de la temporalidad buscada: minutos con `--temporality min`, horas con `h`, sesiones con datos diarios y `d`. `--shorts` busca operaciones cortas. En dirección, `--filter` descarta las operaciones clasificadas como tendencia; sin esa opción invierte su dirección.

```python
import keys
from find_best import opti_main
from find_best_dir import opti_dir

keys.temp = "d"
keys.candles_min, keys.candles = 1, 5
keys.lookbacks_min, keys.lookbacks = 2, 110
keys.calls = 60

best = opti_main("precios.csv", engie="fm")
params, metrics = opti_dir("precios.csv", n_trials=100, ma_seed=best)
print(params, metrics)
```

`opti_main` devuelve `[método, temporalidad, período]` e imprime sus métricas si `verbose=True`. `opti_dir` devuelve parámetros y métricas. Con `filter=None` en esa función se busca también si descartar o invertir las operaciones de tendencia. En muestras con varios activos se debe indicar `asset`.

## Señales y métricas

Se conserva la estrategia de reversión: entrada larga cuando la media cruza por encima del cierre; entrada corta cuando cruza por debajo. El cruce contrario cierra la operación. Se puntúan únicamente pares completos; una entrada sin salida al final de la muestra se excluye. El calentamiento propio de cada media no genera señales. Los indicadores de dirección se consultan en la última vela cerrada disponible a la entrada.

```python
from read_data import read_asset
from use_tecnics import main
from tester import backtest

prices = read_asset("precios.csv")
signals = main("SMA", prices, lookback=20, candle=1, temporality="d")
metrics = backtest(signals, prices)
print(metrics._asdict())
```

Las métricas son:

| Campo | Cálculo |
| --- | --- |
| `hr` | Fracción de operaciones con retorno estrictamente positivo |
| `rr` | Ganancia media / magnitud de pérdida media |
| `pr` | Suma de ganancias / magnitud de suma de pérdidas |
| `tr` | Número de operaciones cerradas |
| `mae` | Media de excursión adversa máxima por operación, en porcentaje |
| `sqn` | `sqrt(min(tr, 100)) * media(retornos) / desviación muestral` |

El retorno largo es `salida / entrada - 1`; en cortos se invierte su signo. Los retornos no se capitalizan. Con menos de dos observaciones o desviación cero, SQN es cero. Para ganancias sin pérdidas, los ratios usan denominador mínimo `1e-12`; así son positivos y finitos. Sin ganancias, ambos ratios son cero. Los ceros cuentan en `tr` y en el denominador de `hr`, pero no como ganancias ni pérdidas.

El MAE excluye el rango anterior al cierre de entrada e incluye el rango hasta la salida. Con bid/ask usa mínimos bid para largos y máximos ask para cortos, e incluye la pérdida inmediata por spread. Al mezclar operaciones normales e invertidas se pondera por su cantidad.

El fitness de `find_best.py` combina estas métricas; es un criterio de clasificación definido por el proyecto. La búsqueda de dirección compara cada combinación de media con esa misma combinación sin filtro, usando las mismas señales originales y el mismo lado largo/corto.

Si `B` es el fitness base y `F` el fitness con dirección, calcula:

```text
mejora = (F - B) / (F + B)
objetivo = F * ((1 - peso) + peso * mejora)
```

Cuando ambos fitness son cero, mejora y objetivo son cero. La mejora está entre −1 y 1; no es un porcentaje de rentabilidad ni una prueba estadística. El cálculo evita dividir por una base casi cero y se implementa con escalado para evitar desbordamiento. El factor de mejora no puede aumentar el objetivo por encima de `F`, de modo que una calidad final muy baja no obtiene una puntuación enorme por su mejora relativa. Esto es un equilibrio de preferencias, no una garantía de calidad mínima: si todas las candidatas son malas, la búsqueda sigue devolviendo la mejor según este criterio.

El peso predeterminado es `0.75`: el objetivo es `F * (0.25 + 0.75 * mejora)`. Si no hay cambio, conserva `0.25 * F`. Una mejora recibe una bonificación frente a ese nivel; una degradación reduce la puntuación y puede volverla negativa. Una calidad final cero siempre produce objetivo cero. Con peso cero se recupera el objetivo anterior basado únicamente en calidad final; con peso uno se exige mejora para obtener un objetivo positivo. El peso no representa probabilidades ni una contribución porcentual garantizada al ranking.

```bash
python test.py precios.csv --direction --calls 100 --improvement-weight 0.75
```

En Python se configura con `opti_dir(..., improvement_weight=0.75)`, dentro de `[0, 1]`. Se conserva la búsqueda conjunta de método, temporalidad, período e indicador de dirección; `ma_seed` solo propone ensayos iniciales. Cada referencia base se calcula una sola vez por combinación de media.

Se conserva la penalización de estabilidad: se maximiza la media del **nuevo objetivo combinado** en las configuraciones vecinas de dirección menos 1,5 desviaciones estándar. Los límites de las vecinas coinciden con los de búsqueda. Los optimizadores usan semilla 42.

La salida de dirección incluye `base_fitness`, `final_fitness`, `fitness_gain` (diferencia absoluta), `improvement_factor` (mejora acotada), `combined_fitness` (objetivo central), `improvement_weight` e `improved` (si el fitness central supera al base). Las métricas `base_hr`, `base_rr`, etc. describen la media sin dirección; `hr`, `rr`, etc. describen el resultado con dirección. `score` es el objetivo combinado penalizado por estabilidad, y `std_score` su dispersión entre vecinos. Un `score` positivo no garantiza `improved=True`; siempre deben consultarse ambos resultados para comprobar qué aportó el filtro.

Las señales se valoran al cierre que genera el cruce. Con OHLC se usa close; con bid/ask los indicadores usan mid, compras a ask y ventas a bid. Al invertir una operación se recalculan ambos precios. No se incluyen comisiones ni deslizamiento. Estos son supuestos del backtest para comparar parámetros, no una simulación de ejecución real.

## Datos e indicadores

`read_asset` acepta OHLC, close o bid/ask en DataFrame, Series con fechas, registros, arrays estructurados, xarray y archivos CSV/Parquet/JSON/Feather/NetCDF. Valida fechas, precios positivos y finitos, consistencia OHLC y ask mayor o igual que bid. Normaliza fechas a UTC sin zona horaria. Con solo close se replica ese precio en OHLC: el MAE refleja únicamente cierres.

Las velas intradía se cierran por la derecha y usan una rejilla fija (`keys.origin`, por defecto `epoch`). Se excluyen velas cuyo cierre todavía no esté disponible. Los datos diarios se agrupan por sesiones completas disponibles. `is_liquid` impide entradas nuevas cuando vale cero.

Las nueve medias usan TA-Lib: SMA, EMA, WMA, DEMA, TEMA, TRIMA, KAMA, T3 y MIDPOINT. Los filtros están en `direction_methods.py`: KEF, Hurst R/S, cociente de varianzas, ADX de Wilder, razón de volatilidad, entropía de Shannon, expansión ATR y ruptura Keltner. El cociente de varianzas sin suficientes observaciones o con varianza nula queda indefinido y no activa tendencia. Hurst y cociente de varianzas son indicadores de régimen, no pruebas de significancia.

## Verificación

```bash
opt/bin/python -m unittest discover -s tests -v
```

Las pruebas cubren métricas contra cálculos manuales, largos/cortos, spread, inversión de dirección, MAE, casos sin operaciones/ganancias/pérdidas, causalidad de todas las medias y filtros, formación de velas, formatos de datos, invariancia frente a escala de precios y optimizadores. Se rechazan pares incompletos, precios inválidos y señales desordenadas o con dirección incompatible.

La selección se realiza sobre la muestra suministrada. Las métricas de esa misma muestra sirven para comparar configuraciones; no miden por sí solas su capacidad de generalizar a datos nuevos.
