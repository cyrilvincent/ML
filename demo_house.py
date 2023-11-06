import pandas as pd
import matplotlib.pyplot as plt

dataframe = pd.read_csv("data/house/house.csv")
print(dataframe.describe())

x = dataframe.surface
y = dataframe.loyer

plt.scatter(x, y)
plt.show()

