import read_data
import new_pipeline
import pandas as pd

activo = pd.read_parquet("Data/EURUSD_TbT_202503310005_202603301319.parquet")
print(activo)
