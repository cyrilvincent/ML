import pandas as pd

dataframe = pd.read_csv("data/heartdisease/data_with_nan.csv", na_values=".")
dataframe = dataframe.drop("slope", axis=1) # idem pour ca et thal
dataframe = dataframe.dropna()

# Le nombre de ligne dataframe.values.shape[0]
# Sauvegarder le dataframe nettoyé dans dataclean.csv
# Trouver une corrélation statistique sur chol, thalach