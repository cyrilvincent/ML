import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

print("Hello World")
print(pd.__version__)

dataframe = pd.read_csv("data/house/house.csv")
dataframe["loyer_m2"] = dataframe.loyer / dataframe.surface
print(dataframe.describe())
avg = np.mean(dataframe.loyer)
std = np.std(dataframe.loyer)
print(avg, std)


dataframe = dataframe[dataframe.surface < 200]
dataframe = dataframe[dataframe.loyer < avg + 3 * std]
plt.scatter(dataframe.surface, dataframe.loyer)
plt.plot(np.arange(200), [avg] * 200, color="red")
plt.show()
