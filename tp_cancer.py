import pandas as pd
import numpy as np
import sklearn.neighbors as n
import sklearn.model_selection as ms
import sklearn.preprocessing as pp

dataframe = pd.read_csv("data/breast-cancer/data.csv", index_col="id")
y = dataframe.diagnosis
x = dataframe.drop("diagnosis", axis=1)

xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, test_size=0.2, train_size=0.8)

# KNN avec k=3
# tester avec différentes valeurs de k
# RobustScaler
# scaler = model

model = n.KNeighborsClassifier(n_neighbors=3)

model.fit(xtrain, ytrain)
score = model.score(xtest, ytest)
print(score)


