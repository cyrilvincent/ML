import pandas as pd
import sklearn.linear_model as sklm

data = pd.read_csv('data/house/house.csv')
regr = sklm.LinearRegression()
x = data["surface"].values.reshape(-1, 1)
y = data["loyer"]

import sklearn.model_selection as ms
import numpy as np
xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, train_size=0.8, test_size=0.2)

import sklearn.preprocessing as pp
import sklearn.pipeline as pipe
model = pipe.make_pipeline(pp.PolynomialFeatures(3), sklm.Ridge())
model.fit(xtrain,ytrain)

predict = model.predict(xtest)

print(model.score(xtest, ytest))

import matplotlib.pyplot as plt
plt.scatter(xtrain, ytrain)
plt.scatter(xtest, predict)
print(model)
plt.show()