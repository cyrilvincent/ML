import pandas as pd

print(pd.__version__)

dataframe = pd.read_csv("data/house/house.csv")
print(dataframe)
print(dataframe.describe())
dataframe["loyer_m2"] = dataframe.loyer / dataframe["surface"]
dataframe.to_excel("data/house/house.xlsx", index=False)  # openpyxl






