# Dans heartdisease data_with_nan.csv & data_cleaned_up.csv
# Le contenu du fichier est expliqué dans index.txt
# Essayer de comprendre le fichier
# Essayer de faire les stats appropriées sur data_cleaned_up.csv

import pandas as pd

dataframe = pd.read_csv("data/heartdisease/data_with_nan.csv", na_values=".")
dataframe_clean = dataframe.drop("slope", axis=1).drop("ca", axis=1).drop("thal", axis=1)
dataframe_clean = dataframe_clean.dropna()
print(dataframe_clean)

dataframe_clean.to_csv("data/heartdisease/dataclean.csv")
print(dataframe_clean.corr())