import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

print(pd.__version__)

dataframe = pd.read_csv("data/house/house.csv", na_values=".")
# dataframe = pd.read_excel("data/house/house.xlsx")
print(dataframe.describe())
print(dataframe)
print(dataframe["surface"])
print(dataframe.loyer)
dataframe.to_html("data/house/house.html")

loyer_par_m2 = dataframe.loyer / dataframe.surface
print(loyer_par_m2.describe())


plt.scatter(dataframe.surface, dataframe.loyer)
plt.show()

# Mini TP
# Lire house.csv avec Pandas
# Afficher le describe
# Calculer loyer par m² => describe
# Afficher dans matplotlib