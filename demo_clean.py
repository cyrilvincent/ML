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
mean_chol = dataframe["chol"].mean()
std_chol = dataframe["chol"].std()
# dataframe.chol == dataframe["chol"]
print(dataframe["ca"].isnull().sum() / 294)

print(dataframe.isnull().sum())
dataframe = dataframe.drop(["slope", "thal", "ca"], axis=1)

l = []
for col in dataframe.columns:
    l.append(col)
print(l)

dataframe = dataframe.fillna({"chol": np.round(mean_chol + (np.random.rand() - 0.5) * std_chol, 0)})
dataframe = dataframe.dropna()
print(dataframe.isnull().sum())

dataframe.to_csv("data/heartdisease/dataclean.csv")

dataframe["chol"].hist()
plt.show()

# TP
# Charger pd.read_csv("data/heartdisease/data_with_nan.csv", na_values=["", "."])
# Compter les isnull
# Pour les colonnes où isnull > 100 => effacer les colonnes
# Pour les colonnes où 10 < chol < 100 => remplacer la valeur par moyenne + rand * std
# Pour les autres colonnes, effacer les lignes avec des nan => dropna()
# Sauvegarder le tableau dans dataclean.csv
