# Charger house.csv avec pandas
# Faire un describe
# Créer la colonne loyer_m2
# Sauvegarder dans le format de votre choix

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xlrd
import sqlite3

dataframe = pd.read_csv("data/house/house.csv", delimiter=",")

dataframe["surface"] = dataframe["surface"].astype(np.float64) # Changement de type
dataframe["loyer_m2"] = dataframe["loyer"] / dataframe["surface"]
# dataframe.to_excel("data/house/house.xlsx", index=False)
dataframe.to_html("data/house/house.html", index=False)

dataframe = dataframe[dataframe["surface"] < 200]
plt.scatter(dataframe["surface"], dataframe["loyer"])

print(dataframe.dtypes)
print(dataframe)
# print(dataframe.values) # convert to numpy
print(dataframe.describe())
plt.show()

# Filtrer la surfaces < 200
# Calculer la mean et la std des loyers
# Enlever les points > mean + 3*std
# Reafficher les données filtrées
# Save