import sklearn
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import sklearn.pipeline as pipe
import sklearn.preprocessing as pp
import sklearn.model_selection as ms
import sklearn.neighbors as n
import numpy as np

df = pd.read_csv("data/breast-cancer/data.csv")
print(df.describe())

np.random.seed(42)

y = df["diagnosis"]
x = df.drop(["diagnosis", "id"], axis=1)

xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, train_size=0.8, test_size=0.2)

print(xtest.shape)

for k in range(3, 15, 2):
    model = n.KNeighborsClassifier(n_neighbors=k)
    model.fit(xtrain, ytrain)
    print(f"Train score for k={k}: {model.score(xtrain, ytrain):.2f}")
    print(f"Test score for k={k}: {model.score(xtest, ytest):.2f}")



