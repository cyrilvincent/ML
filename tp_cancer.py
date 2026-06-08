import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import sklearn.preprocessing as pp
import sklearn.pipeline as pipe
import sklearn.model_selection as ms
import numpy as np
import sklearn.neighbors as n

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 500)

df = pd.read_csv("data/breast-cancer/data.csv")
print(df.describe())

# kNN
# Trouver la meilleur valeur de k
