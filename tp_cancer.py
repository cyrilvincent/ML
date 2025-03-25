import pandas as pd
import numpy as np
import sklearn.neighbors as n

dataframe = pd.read_csv("data/breast-cancer/data.csv", index_col="id")
y = dataframe.diagnosis
x = dataframe.drop("diagnosis", axis=1)

# KNN avec k=3
# tester avec différentes valeurs de k
