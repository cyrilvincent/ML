import pandas as pd
import matplotlib.pyplot as plt

print("Hello")
print(pd.__version__)

dataframe = pd.read_csv("data/house/house.csv")
dataframe = dataframe[dataframe.surface < 200]
print(dataframe.describe())

res = dataframe.loyer / dataframe.surface
print(res.describe())

plt.scatter(dataframe.surface, dataframe.loyer)
plt.show()