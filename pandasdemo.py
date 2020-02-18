import pandas as pd
import numpy as np

dataframe = pd.read_csv("data/house/house.csv")

res = dataframe.loyer / dataframe.surface


dataframe = dataframe[dataframe.surface < 300]

mean = np.mean(res)
std = np.std(res)

dataframe = dataframe[abs(dataframe.loyer / dataframe.surface - mean) < 3 * std]


import scipy.stats as stats
slope, intercept, rvalue, pvalue, error = stats.linregress(dataframe.surface, dataframe.loyer)
print(slope, intercept, rvalue, pvalue, error)

import matplotlib.pyplot as plt
plt.scatter(dataframe.surface, dataframe.loyer)
plt.plot(dataframe.surface, slope * dataframe.surface + intercept, color="red")
plt.show()




