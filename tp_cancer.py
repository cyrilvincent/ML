# Charger data/breast-cancer/data.csv
# pd.read_csv("data/breast-cancer/data.csv", index_col="id")
# seed + train test + knn + fit + score + predict + conclusion
# tester différentes valeurs de k

import pandas as pd
import numpy as np
import sklearn.neighbors as nn
import sklearn.model_selection as ms

dataframe = pd.read_csv("data/breast-cancer/data.csv", index_col="id")
print(dataframe.describe())

np.random.seed(42)

y = dataframe["diagnosis"]
x = dataframe.drop("diagnosis", axis=1)

xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, train_size=0.8, test_size=0.2)

for k in range(3,12,2):
    model = nn.KNeighborsClassifier(n_neighbors=k)

    model.fit(xtrain, ytrain)

    scoretrain = model.score(xtrain, ytrain)
    scoretest = model.score(xtest, ytest)

    print(scoretrain, scoretest)

ypred = model.predict(xtest)
print(ypred)
