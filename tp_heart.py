# Charger data/heartdisease/data_with_nan.csv
# Dropper les colonnes slope,ca,thal
# Dropper les lignes avec des nan : dropna()
# Sauvegarder dans dataclean.csv : to_csv
# Créer 2 datasets : ok pour les patients sains : num = 0
#                    ko pour les malades : num = 1
# Afficher les moyennes et écart types (mean, std) des 2 datasets pour la colonne chol
# Afficher les corrélations des datasets : corr()

import pandas as pd
import numpy as np

dataframe = pd.read_csv("data/heartdisease/data_with_nan.csv", na_values='.')
dataframe = dataframe.drop("slope", axis=1).drop("ca", 1).drop("thal", 1)
dataframe = dataframe.dropna()
dataframe.to_csv("data/heartdisease/dataclean.csv")
ok = dataframe[dataframe.num == 0]
ko = dataframe[dataframe.num == 1]
print(np.mean(ok.chol), np.std(ok.chol))
print(np.mean(ko.chol), np.std(ko.chol))
print(ko.corr())