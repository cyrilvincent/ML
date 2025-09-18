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
dataframe = pd.read_csv("data/heartdisease/data_cleaned_up.csv")

# Déterminer x et y
y = dataframe["num"]
x = dataframe.drop("num", axis=1)

# 3 Model
# model = lm.LinearRegression()
# f(x) = ax + b; a = slope; b = intersection pour x=0;
# a et b soit les monômes du polynôme de dégré 1; a et b sont des poids
# for degree in range(2,5):
# degree = 2
# model = pipe.make_pipeline(pp.PolynomialFeatures(degree), lm.Ridge())
# f(x) = ax² + bx + c

for k in range(1,10):
    model = nb.KNeighborsClassifier(n_neighbors=k)

    # 4 Apprentissage supervisé car y est connu
    model.fit(x, y)

    # 5 Score
    score = model.score(x, y)
    print("Score", k, score)




