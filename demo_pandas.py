import pandas as pd
import numpy as np

print(pd.__version__)

dataframe = pd.read_csv("data/house/house.csv")
print(dataframe)
print(dataframe.describe())

dataframe["loyer_m2"] = dataframe.loyer / dataframe["surface"]
dataframe.to_excel("data/house/house.xlsx", index=False)  # openpyxl

loyer_moyen = np.mean(dataframe.loyer)
loyer_std = np.std(dataframe.loyer)

print(f"Loyer moyen: {loyer_moyen}, ecart type: {loyer_std}")
print(dataframe[dataframe.surface > 200])







