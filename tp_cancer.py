import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn.pipeline as pipe
import sklearn.preprocessing as pp
import sklearn.model_selection as ms
import sklearn.neighbors as n

df = pd.read_csv("data/breast-cancer/data.csv")
print(df.describe())

y = df["diagnosis"]
x = df.drop(["id", "diagnosis"], axis=1)

xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, train_size=0.8, test_size=0.2, random_state=42)

scaler = pp.RobustScaler()
scaler.fit(xtrain)
xtrain = scaler.transform(xtrain)
xtest = scaler.transform(xtest)

for k in range(3, 12):
    model = n.KNeighborsClassifier(n_neighbors=k)
    model.fit(xtrain, ytrain)

    print("Training Score", model.score(xtrain, ytrain))
    print("Testing Score", model.score(xtest, ytest))

