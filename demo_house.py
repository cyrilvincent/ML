import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataframe = pd.read_csv("data/house/house.csv")
print(dataframe)
loyers_per_m2 = dataframe["loyer"] / dataframe.surface
print(np.mean(loyers_per_m2), np.std(loyers_per_m2), np.median(loyers_per_m2))
print(dataframe.describe())

plt.scatter(dataframe.surface, dataframe.loyer)
plt.show()

dataframe.surface.plot.hist()
plt.show()