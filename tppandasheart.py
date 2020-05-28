# Charger le fichier data/heartdisease/data_with_nan.csv pd.read_csv(filename,na_values=".")
# Afficher dataframe et noter les valeurs de l'avant dernière colonne
# Effacer les colonnes slope,ca,thal : dataframe.drop(nomcol,1)
# Nettoyer le fichier en remplacant les nan par 0 : fillna
# Calculer la moyenne et l'écart type de chol
# Refaire en effacant les nan : dropna
# ReCalculer la moyenne et l'écart type de chol

import pandas as pd
import numpy as np

dataset = pd.read_csv("data/heartdisease/data_with_nan.csv", na_values=".")
dataset = dataset.drop('slope',1)
dataset = dataset.drop('ca',1)
dataset = dataset.drop('thal',1)
dataset2 = dataset.fillna(0)
print(dataset2)
print(np.mean(dataset2.chol), np.mean(dataset2.age))
dataset3 = dataset.dropna()
print(np.mean(dataset3.chol), np.mean(dataset3.age))
