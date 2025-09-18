import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xlrd
import sqlite3

# dataframe = pd.read_csv("data/house/house.csv", delimiter=",")
# dataframe = pd.read_excel("data/house/house.xlsx")
with sqlite3.connect("data/house/house.db3") as conn:
    dataframe = pd.read_sql("select * from house", conn)

plt.scatter(dataframe["surface"], dataframe["loyer"])
dataframe["surface"] = dataframe["surface"].astype(np.float64) # Changement de type
dataframe["loyer_m2"] = dataframe["loyer"] / dataframe["surface"]
# dataframe.to_excel("data/house/house.xlsx", index=False)
dataframe.to_html("data/house/house.html", index=False)
print(dataframe.dtypes)
print(dataframe)
# print(dataframe.values) # convert to numpy
print(dataframe.describe())
plt.show()