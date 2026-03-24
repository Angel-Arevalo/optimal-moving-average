import read_data
import new_pipeline
import pandas as pd

activo = pd.read_parquet("Data/EURUSD_2019-2025.parquet")
activo.index = pd.to_datetime(activo.index)

activo = read_data.ohlc_form(activo, "5min")

# ✅ CORRECTO
df_alpha, horizon = new_pipeline.run_pipeline(activo)

df_regime = new_pipeline.apply_regime_filter(df_alpha)

df_final, metrics = new_pipeline.run_execution_layer(df_regime, horizon)

print("--- MÉTRICAS DEL BACKTEST ---")
for metrica, valor in metrics.items():
    if metrica in ['total_return', 'cagr', 'max_dd']:
        print(f"{metrica.upper()}: {valor * 100:.2f}%")
    else:
        print(f"{metrica.upper()}: {valor:.2f}")
