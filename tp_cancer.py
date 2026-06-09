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

y = df["diagnosis"]
x = df.drop(["diagnosis", "id"], axis=1)

np.random.seed(42)

xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, train_size=0.8, test_size=0.2)

scaler = pp.RobustScaler()
scaler.fit(xtrain)
xtrain = scaler.transform(xtrain)
xtest = scaler.transform(xtest)

for k in range(3, 12):
    model = n.KNeighborsClassifier(n_neighbors=k, )
    model.fit(xtrain, ytrain)

    print(k)
    print(f"Train score: {model.score(xtrain, ytrain):.1f}")
    print(f"Test score: {model.score(xtest, ytest):.1f}")

ypred = model.predict(xtest)
print(ypred)

# scaler.inverse_transform(xtrain)


