import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("Hello World!")
print(np.__version__)
print(pd.__version__)

v12 = np.arange(12)
print(v12)
print(v12.shape)
mat34 = v12.reshape(-1, 4)
print(mat34)
print(mat34.shape)
print(np.tanh(mat34))

dataframe = pd.read_csv("data/house/house.csv")
print(dataframe)

plt.scatter(dataframe["surface"], dataframe["loyer"])
plt.show()

dataframe_heart = pd.read_csv("data/heartdisease/data_cleaned_up.csv")
print(dataframe_heart.describe().T)
dataframe.hist()
plt.show()
