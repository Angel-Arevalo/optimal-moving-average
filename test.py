import read_data
import new_pipeline
import pandas as pd
from find_best import opti_main
import keys
keys.calls = 10
keys.methods = {"SMA"}
opti_main("EURUSD_TbT_202503310005_202603301319.parquet", True)
