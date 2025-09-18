import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = np.nan  # Not a number
v1 = np.array([1,2,np.nan,4])
print(v1)
print(np.sum(v1))
print(np.nansum(v1))

v2 = np.array([1,2,np.inf,4])
print(v2.sum())
print(v2 / 2)

dataframe = pd.read_csv("data/heartdisease/data_with_nan.csv", na_values=["", "."])
print(dataframe)
print(dataframe["chol"].values.mean())
# dataframe.chol == dataframe["chol"]
print(dataframe["ca"].isnull().sum() / 294)
dataframe = dataframe.drop(["ca"], axis=1)

print(dataframe["chol"].isnull().sum() / 294)
print(dataframe.isnull().sum())
dataframe["chol"].hist()
plt.show()

# TP
# Charger pd.read_csv("data/heartdisease/data_with_nan.csv", na_values=["", "."])
# Compter les isnull
# Pour les colonnes où isnull > 100 => effacer les colonnes
# Pour les colonnes où 10 < chol < 100 => remplacer la valeur par moyenne + rand * std
# Pour les autres colonnes, effacer les lignes avec des nan => dropna()
# Sauvegarder le tableau dans dataclean.csv
