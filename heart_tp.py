import pandas as pd
import matplotlib.pyplot as plt

dataframe = pd.read_csv("data/heartdisease/data_with_nan.csv", na_values=".")
dataframe = dataframe.drop("slope", axis=1).drop("ca", axis=1).drop("thal", axis=1) # idem pour ca et thal
size_before = dataframe.values.shape[0]
dataframe = dataframe.dropna()
size_after = dataframe.values.shape[0]
print(size_before, size_after)
corr = dataframe.corr()
print(corr)
plt.matshow(corr)
plt.show()

