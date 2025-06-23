import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


print("Hello World")
print(sys.version)
print(pd.__version__)

dataframe = pd.read_csv("data/house/house.csv")
dataframe = dataframe[dataframe.surface < 180]
print(dataframe.describe())
result = dataframe["loyer"] / dataframe.surface
print(np.mean(result))
print(np.median(dataframe.surface), np.quantile(dataframe.surface, [0.01, 0.1, 0.25, 0.75, 0.9, 0.99]))

dataframe.hist(bins=20)
plt.show()
plt.scatter(dataframe.surface, dataframe.loyer)
plt.show()