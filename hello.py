import pandas as pd
import matplotlib.pyplot as plt

print("Hello World")
print(pd.__version__)

dataframe = pd.read_csv("data/house/house.csv")
print(dataframe.describe())

plt.scatter(dataframe.surface, dataframe.loyer)
plt.show()
