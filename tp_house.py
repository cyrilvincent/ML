import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print(pd.__version__)

dataframe = pd.read_csv("data/house/house.csv")
dataframe["loyer_m2"] = dataframe.loyer / dataframe.surface
print(dataframe)

# Afficher en x les surfaces et en y les loyers dans un scatter
# Que peut on en déduire?
# Filtrer les surfaces < 180
# Reafficher
# Save
# dataframe.hist(bins=5)


dataframe2 = dataframe[dataframe.surface < 180]
plt.title("Surface / Loyer à Paris")
plt.scatter(dataframe2.surface, dataframe2.loyer, label="Surface / Loyer")
plt.xlabel("Surface")
plt.ylabel("Loyer")
plt.savefig("data/house/house.png")

dataframe2.hist(bins=10)
plt.show()
