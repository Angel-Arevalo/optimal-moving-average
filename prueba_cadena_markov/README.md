# Filtro causal de regímenes para las entradas LONG

Este directorio añade un experimento integrado con el código del repositorio. Conserva el optimizador y las condiciones de cruce. La evaluación y la calibración no heredan sus liquidaciones semanales: sólo cierran por un cruce de salida real. El resultado de la inspección previa está en [AUDITORIA.md](AUDITORIA.md); las métricas ejecutadas y su interpretación están en [RESULTADOS.md](RESULTADOS.md).

## Archivos implementados

| Archivo | Responsabilidad |
|---|---|
| `config.py` | Parámetros inmutables de observaciones, HMM, ventanas y contabilidad |
| `market.py` | Adaptar el optimizador original, restaurar sus globals y alinear medias/señales por disponibilidad |
| `model.py` | Ajuste, controles numéricos, forward causal, streaming y persistencia |
| `economics.py` | Seleccionar estados con bootstrap, estadísticas y equity incluyendo posiciones abiertas |
| `execution.py` | Mantener posiciones entre ventanas, congelar la media de entrada y ejecutar sólo salidas por cruce |
| `walk_forward.py` | Cuatro bloques temporales, caché, A/B, sensibilidad histórica y exportación |
| `download.py` | Descargar y registrar procedencia de OHLC Quantiacs mediante el entorno uv |
| `__main__.py` | Comando reproducible de comparación |
| `tests/` | Pruebas unitarias, sintéticas y de integración temporal |
| `pyproject.toml`, `uv.lock` | Dependencias y versiones aisladas dentro de esta carpeta |

No se editó ningún archivo fuente original de la raíz.

## Ejecutar desde la raíz del repositorio

```bash
uv sync --project prueba_cadena_markov --locked

# Quantiacs oficial, BTC horario, final exclusivo. Descarga real, no datos sintéticos.
uv run --project prueba_cadena_markov python -m prueba_cadena_markov.download \
  --asset BTC --start 2021-12-27 --end 2024-02-26

# A/B fijo: ocho semanas fuera de muestra; C/D solamente dentro de calibración.
uv run --project prueba_cadena_markov python -m prueba_cadena_markov --experiments

uv run --project prueba_cadena_markov pytest prueba_cadena_markov/tests -q \
  --junitxml=prueba_cadena_markov/results/tests.xml
```

Dependencias y lock propios del experimento; `uv` crea `prueba_cadena_markov/.venv`. Se reutilizan los módulos Python de la raíz, sin copiarlos. `qnt` queda fijado a un commit en `uv.lock`. El descargador respeta `API_KEY` si existe y usa el acceso público `default` en caso contrario. No registra credenciales. Guarda Parquet, procedencia y SHA256 en `data/`; caché Quantiacs en `cache/quantiacs/`. Datos, entornos y modelos binarios se excluyen de Git; los CSV de resultados, metadatos y tests se conservan.

La [API oficial de Quantiacs](https://quantiacs.com/documentation/en/reference/data_load_functions.html) ofrece OHLC horario mediante `qnt.data.crypto.load_data`. No se confunde este conjunto con el diario `cryptofutures`. Los timestamps se interpretan como inicios de velas horarias UTC, cuya disponibilidad es una hora después.

Para otra ejecución, usar una carpeta nueva y un período que cubra las cuatro fases:

```bash
uv run --project prueba_cadena_markov python -m prueba_cadena_markov \
  --data prueba_cadena_markov/data/BTC_hourly.parquet \
  --start 2024-01-01 --weeks 8 --calls 20 --candle-max 6 --lookback-max 40 \
  --optimization-weeks 26 --fit-weeks 26 --calibration-weeks 52 \
  --n-states 3 --threshold 0.70 --min-trades 30 \
  --capital 100000 --units 1 --fee-bps 0 \
  --output prueba_cadena_markov/results/otra_ejecucion
```

No cambiar parámetros después de mirar el test y seguir llamándolo fuera de muestra. Para nuevos ajustes se necesita otra evaluación futura. El comando compara siempre A/B; no hace trading real.

## Observaciones y disponibilidad

La frecuencia HMM es la resolución fuente, **una hora para este experimento**, independiente del `candle` elegido para la media. Las señales siguen apareciendo únicamente en el cierre del timeframe original elegido. Se calcula la misma media TA-Lib que usa la estrategia y se alinea por último cierre disponible (`ffill` de **medias ya cerradas**, nunca de precios futuros).

En este orden, compartido por entrenamiento e inferencia:

| Variable | Cálculo |
|---|---|
| `distance_to_ma` | `log(P) - log(MA)` |
| `distance_change` | `X - X.shift(1)` |
| `ma_slope` | `(log(MA) - log(MA).shift(k))/k`, `k=3` |
| `realized_volatility` | `sqrt(sum(log(P/P.shift(1))**2, últimas n observaciones))`, `n=20` |
| `efficiency_ratio` | `abs(P-P.shift(n))/sum(abs(P.diff()), n)`, `n=20`; denominador cero → cero |

`FeatureConfig` configura los períodos. No hay `backfill`, interpolación, centrado de ventanas ni diferencias de precio bruto como feature. La rejilla regular incluye huecos de datos como NaN. Ventanas incompletas, NaN, infinitos y medias no positivas invalidan la observación y bloquean entradas. Una media DEMA/TEMA puede no ser positiva aunque lo sean los precios; también se valida. OHLC fuente no positivo se rechaza de forma explícita porque no permite esta definición logarítmica. No se inventa una transformación para instrumentos de precios negativos.

## Ajuste y probabilidades

`MarkovRegimeFilter.fit(features, available_by=...)` ajusta `StandardScaler` exclusivamente en el bloque fit y entrena `GaussianHMM`: tres estados, covarianza diagonal, 200 iteraciones, tolerancia `1e-4`, semilla 42. Los valores se configuran en `MarkovConfig`. Se comprueban tamaño del conjunto, ocupación de estados, parámetros finitos, covarianzas, condición numérica y transición estocástica. Las secuencias de ajuste se cortan en huecos de datos.

Se exige que la mejora de log-verosimilitud alcance la tolerancia; llegar al límite de iteraciones **no basta**. Esto es más estricto que `monitor_.converged` de hmmlearn, que también considera terminado el ajuste al alcanzar `n_iter`. Un ajuste inválido bloquea todas las entradas de esa ventana; no se aumenta automáticamente el presupuesto mirando su test.

La inferencia implementa explícitamente:

```text
log_alpha[j] = log_emission(Y_t, j)
             + logsumexp(log_alpha_previous[i] + log_transition[i,j], i)
log_alpha -= logsumexp(log_alpha)
```

Sólo devuelve `P(S_t=j | Y_1,...,Y_t, parámetros históricos congelados)`. No usa `predict`, Viterbi ni `predict_proba`: [el código de hmmlearn](https://github.com/hmmlearn/hmmlearn/blob/0.3.3/src/hmmlearn/base.py) combina forward/backward en sus posteriores. Se prueba la equivalencia con Bayes directo y con el filtrado observación por observación. En un hueco se propaga la transición sin emisión, se devuelve NaN y se bloquea la entrada.

`predict_filtered_proba` no modifica el modelo. Para streaming usar `model.stream().update(frame_de_una_fila)`, en orden cronológico. Hay que inicializarlo recorriendo el historial disponible si se desea reproducir exactamente el posterior de un backtest; iniciar un stream vacío usa la distribución inicial del modelo. También se puede pasar un posterior inicial a la inferencia batch. No se debe usar el último posterior de test para inferir una fecha anterior.

## Qué estados se pueden operar

El ID de estado no tiene significado económico fijo y puede permutarse entre semanas. Se conservan todas las probabilidades a la entrada y se asigna el trade al `argmax` **de ese instante**. Se usan operaciones cerradas por cruces naturales, sin liquidaciones por calendario, con su precio de salida y costos; se excluyen trades que entran antes de calibración o salen después de ella.

Un estado favorable necesita al menos 30 operaciones, retorno neto medio positivo y un límite inferior de esperanza estrictamente positivo. Se estima con 2.000 bootstrap circulares de bloques de cinco operaciones consecutivas asignadas a ese estado, semilla fija. El cuantil unilateral se ajusta por el número de estados (Bonferroni, confianza nominal 95%). Esta estimación conserva dependencia local entre operaciones, pero no demuestra independencia, estabilidad temporal ni controla todas las decisiones de investigación.

Si ningún estado cumple, se abstiene. Nunca se fuerza un ganador porque el HMM haya encontrado componentes. La probabilidad favorable es la **suma** de probabilidades de los estados habilitados. Se permite una entrada original cuando esa suma es ≥0,70. `current_regime` es el estado más probable; `regime_confidence`, su probabilidad. La suma favorable puede superar el umbral aunque ninguno de sus estados individualmente lo haga.

**Una probabilidad de régimen no es una probabilidad de ganar el trade. Un HMM no demuestra reversión rentable.**

## Separación temporal y fronteras

Cada semana de test tiene:

```text
26 semanas optimización MA → 26 semanas fit HMM → 52 semanas calibración → 1 semana test
```

Todo se desplaza una semana; el modelo se reentrena al comienzo de cada ventana, nunca en cada vela. La media se selecciona **antes** de las observaciones usadas para fit. Su historia anterior sólo sirve de warmup. No hay selección retrospectiva de una media con el rendimiento de su fit/calibración/test.

Esta versión sacrifica frescura de la media para separar los procesos sin reescribir el optimizador ni introducir un walk-forward anidado costoso. El conjunto de medias, operaciones y supuestos de ejecución es idéntico para A/B. Ajuste, calibración y test nunca se mezclan para entrenar un modelo. Una semana ya evaluada podrá pertenecer al pasado de ventanas futuras cuando corresponda a sus bloques; nunca se usa el futuro de la ventana actual.

Las posiciones permanecen abiertas al cambiar de semana o modelo. La media, el timeframe, el lookback y el origen de su historial quedan congelados desde la entrada hasta el cruce de salida; la nueva configuración sólo gobierna entradas posteriores. El HMM de una nueva ventana no afecta una posición existente. Si coincide un cruce de salida de la media antigua con una entrada de la nueva, sólo se ejecuta la salida en ese timestamp.

Al terminar el período tampoco se liquida. Las posiciones pendientes se guardan en `*_open_positions.csv` y se valoran al último precio disponible, sin inventar un precio de salida ni cobrar una comisión de salida no ejecutada. Las métricas de trades cuentan sólo operaciones cerradas; retorno, drawdown, Sharpe y exposición incluyen la posición abierta. Un trade que termina en otra semana mantiene el estado, las probabilidades y la ventana de su entrada (`window`), con la ventana de cierre en `exit_window`.

El optimizador conserva `get_vector_buys`, su objetivo y sus liquidaciones internas. La ejecución continua usa exactamente sus desigualdades de cruce, separadas de esa normalización semanal. `contained_trades` purga explícitamente etiquetas que crucen cualquier frontera de calibración. Los resúmenes `fit_descriptive` son retrospectivos bajo un modelo ajustado con todo ese bloque y **no son resultados fuera de muestra**. Los retornos futuros de una observación se guardan sólo como descriptores dentro de su fase; jamás entran en `FEATURES`.

Caché con hashes de datos, configuración, fecha límite, código y versiones de dependencias. Se guarda junto el HMM, scaler, orden de variables, parámetros de features, estados habilitados y fechas. Los IDs se interpretan por ventana. Los pickle son artefactos locales de confianza; no cargar archivos de terceros.

## Activar y desactivar

El valor de `MarkovConfig.enabled` es `False`. El comando comparativo ejecuta ambos brazos: `Original` acepta toda entrada de la estrategia continua; `HMM` decide cada entrada mediante `model.allow_long_entry`/`model.decision` con el modelo congelado de la ventana.

`ContinuousExecution` mantiene un único flujo de operaciones de referencia. Si el HMM rechaza una entrada, no añade entradas alternativas mientras esa operación de referencia sigue abierta. Así, sus operaciones aceptadas conservan exactamente las salidas de la referencia, incluso al atravesar semanas; no se cambia la comparación por diferencias de ocupación. El brazo filtrado permanece en efectivo durante una operación rechazada.

No abre SHORT, no genera cruces ni cierra por cambios de régimen. Cada entrada candidata conserva aceptación, motivo, estado y probabilidades. Se cuentan cruces de entrada cuando la referencia está libre, sin normalización semanal. El helper `gate_signals(..., use_markov_filter=False/True)` se conserva para compatibilidad con pares ya cerrados; no es el ejecutor continuo ni liquida posiciones pendientes.

## Experimentos y resultados guardados

A es original; B es el HMM fijo 3/0,70. `--experiments` añade C/D: estados 2/3/4 y umbrales 0,55/0,65/0,70/0,75/0,85, en sólo 15 combinaciones por ventana. Reutiliza el modelo para los umbrales. La primera mitad de calibración selecciona estados; la segunda informa sensibilidad y suficiencia de muestra. No reemplaza automáticamente B ni elige una configuración mirando el test. Si no alcanza la muestra mínima, lo registra; prevalece A/B fijo. Las calibraciones de ventanas distintas se solapan: no sumar sus P&L como rendimiento de una cartera ni tratarlas como muestras independientes.

| Archivo | Contenido |
|---|---|
| `comparison.csv` | Retorno total/anualizado, DD, Sharpe, PF, win rate, trades, exposición, señales y probabilidades a la entrada |
| `original_trades.csv`, `hmm_trades.csv` | Trades cerrados por cruce, P&L, duración, probabilidades y ventanas de entrada/salida |
| `*_open_positions.csv` | Entradas todavía abiertas y P&L a mercado; sin salida artificial |
| `*_decisions.csv` | Todas las entradas originales y su aceptación o motivo de rechazo |
| `*_equity.csv` | Equity a cada cierre fuente y cantidad LONG |
| `regimes.csv` | Caracterización por estado **y ventana**, separando fit descriptivo/calibración/test y estrategia |
| `regime_selection.csv` | Muestra por estado, esperanza, límite inferior y decisión histórica |
| `calibration_sensitivity.csv` | C/D sólo en calibración, si se solicita |
| `window_*/` | Probabilidades de evaluación, trades por fase y metadatos del modelo |
| `models/`, `manifest.json` | Modelos congelados y trazabilidad de datos, código, parámetros y versiones |
| `../tests.xml` | Resultado JUnit de tests automatizados |

La distribución de estados de la tabla global es sólo un conteo de IDs locales; **no** implica que el estado 1 de dos ventanas sea el mismo régimen. Para análisis económico usar `regimes.csv` agrupado por ventana.

Equity es mark-to-market a cierres fuente, una unidad fija, capital de referencia configurable. Total return = equity final/capital − 1; annualized return usa días calendario; drawdown incluye el capital inicial como pico; Sharpe usa retornos de días cerrados (la medianoche cierra el día anterior, sin añadir un día inicial de retorno cero), tasa libre de riesgo cero y 365,25 días para cripto. Exposición pondera el tiempo con posición. PF de la cuenta usa P&L monetario; los resúmenes por estado usan retornos porcentuales. PF, win rate y retorno medio por trade usan operaciones cerradas; `Net PnL` incluye P&L de posiciones abiertas, separado en `Closed Trade PnL` y `Open Position PnL`. Una cartera sin trades cerrados puede tener exposición y Sharpe por una posición abierta. Sin pérdidas realizadas, PF es infinito; sin trades cerrados, PF y win rate son indefinidos. Con equity no positiva se omiten Sharpe y retorno anualizado.

## Límites del experimento

- Ocho semanas de un solo activo no prueban robustez; anualizar un intervalo corto amplifica ruido.
- Sin spread ni slippage y con fill al mismo cierre, resultados positivos no son una promesa ejecutable. Costos distintos deben probarse por igual en calibración y test.
- Estados gaussianos diagonales y transiciones homogéneas son aproximaciones; las variables están correlacionadas y el mercado cambia.
- La calibración usa trades naturalmente cerrados y purga los que cruzan sus fronteras. Si las posiciones pendientes tienen una distribución de resultados distinta, esa censura puede sesgar la esperanza estimada; no se inventa una liquidación para obtener su etiqueta. La evaluación incluye su P&L a mercado.
- El bootstrap no elimina sobreajuste del optimizador, comparaciones múltiples entre investigaciones, cambios de régimen ni incertidumbre de muestras cortas.
- Rechazar operaciones reduce exposición; una mejora de drawdown puede provenir sólo de mantener efectivo.
- La media se elige en un bloque más antiguo que la calibración. Esta implementación causal no responde todavía a la variante de reoptimizarla justo antes del test usando medias históricas generadas por un walk-forward anidado.
- El adaptador inicial recibe OHLC de un único activo; no integra ticks bid/ask ni reconstruye spread. La ejecución real continua y calendarios no cripto requieren trabajo adicional.

Las pruebas también verifican posiciones que cruzan fines de semana, cambios de media/HMM, ausencia de liquidación terminal, censura de etiquetas pendientes y comisiones de posiciones abiertas. Cubren forward normalizado, equivalencia online/batch, huecos, prefijos y perturbación del futuro, multi-timeframe, fit/scaler históricos, reproducción con semilla, purga de fronteras, ausencia de estados rentables sin evidencia, entradas LONG sin órdenes nuevas, salidas intactas, equivalencia al desactivar, P&L/costos y funcionamiento del optimizador original.

## Corrección de la evaluación anterior

La primera comparación heredaba los cierres semanales del optimizador. Por instrucción del usuario se separaron ambas cosas. Esa ejecución y su informe se conservan sólo como archivo histórico en `results/btc_ab_weekly_legacy/` y `RESULTADOS_CIERRES_SEMANALES.md`; no describen la estrategia continua actual. También se retiró la exclusión de H5 del espacio del optimizador, que ya no es necesaria para sostener cierres semanales en evaluación.
