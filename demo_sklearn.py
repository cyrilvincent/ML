import sklearn
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.preprocessing as pp
import sklearn.pipeline as pipe
import sklearn.linear_model as lm
import numpy as np
import sklearn.model_selection as ms


print(sklearn.__version__)
np.random.seed(0)

dataframe = pd.read_csv("data/house/house.csv")

y = dataframe.loyer
x = dataframe.surface.values.reshape(-1, 1)

xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, train_size=0.8, test_size=0.2)

# model = LinearRegression()
model = pipe.make_pipeline(pp.PolynomialFeatures(2), lm.Ridge())
model.fit(xtrain, ytrain)
# Interdiction de fit sur xtest et ytest

# print(model.coef_, model.intercept_)
#
# # f(x) = ax + b avec a = coef_ et b = intercept_
# f = lambda x: model.coef_ * x + model.intercept_

y_pred = model.predict(np.arange(400).reshape(-1, 1))

score = model.score(xtrain, ytrain)
print(score)
score = model.score(xtest, ytest)
print(score)

plt.scatter(x, y)
plt.plot(range(400), y_pred, color="red")
# plt.plot(x, f(x), color="green")
plt.show()












































































