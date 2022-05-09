import pandas as pd
import matplotlib.pyplot as plt

print(pd.__version__)

dataframe = pd.read_csv("data/house/house.csv")
dataframe = dataframe[dataframe.surface < 200]
x = dataframe.surface
y = dataframe.loyer

dataframe.to_xml("data/house/house.xml")

print(dataframe)

plt.scatter(x, y)
plt.show()


