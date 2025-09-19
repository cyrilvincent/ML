# Refaire fonctionner dom_sklearn + tp_heart_sklearn
# Refaire tp_heart_sklearn pour le cancer
# y = diagnosis
# x = tout sauf diagnosis et id

# Refaire une regression

import numpy as np
import sklearn
import sklearn.linear_model as lm
import sklearn.preprocessing as pp
import sklearn.pipeline as pipe
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.neighbors as nb
print(sklearn.__version__)

# 1 Pandas DataMart
dataframe = pd.read_csv("data/breast-cancer/data.csv")

# Déterminer x et y
y = dataframe["diagnosis"]
x = dataframe.drop(["diagnosis", "id"], axis=1)

# 3 Model
# model = lm.LinearRegression()
# f(x) = ax + b; a = slope; b = intersection pour x=0;
# a et b soit les monômes du polynôme de dégré 1; a et b sont des poids
# for degree in range(2,5):
degree = 2
# model = pipe.make_pipeline(pp.PolynomialFeatures(degree), lm.Ridge())
# f(x) = ax² + bx + c
model = nb.KNeighborsClassifier(n_neighbors=3)

# 4 Apprentissage supervisé car y est connu
model.fit(x, y)

# 5 Score
score = model.score(x, y)
print("Score", score)

# (x-mean)/std



