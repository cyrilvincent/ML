import pandas as pd
import numpy as np

dataframe = pd.read_csv("data/house/house.csv")
dataframe = dataframe[dataframe.surface < 250]

import sqlite3
with sqlite3.connect("data/house/house.db3") as conn:
    dataframe = pd.read_sql("select * from house", conn)

res = dataframe.loyer / dataframe.surface
print(np.mean(res), np.std(res))