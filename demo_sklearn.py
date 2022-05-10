import sklearn
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.preprocessing as pp
import sklearn.pipeline as pipe
import sklearn.linear_model as lm
import numpy as np

print(sklearn.__version__)

dataframe = pd.read_csv("data/house/house.csv")

y = dataframe.loyer
x = dataframe.surface.values.reshape(-1, 1)

# model = LinearRegression()
model = pipe.make_pipeline(pp.PolynomialFeatures(7), lm.Ridge())
model.fit(x, y)

# print(model.coef_, model.intercept_)
#
# # f(x) = ax + b avec a = coef_ et b = intercept_
# f = lambda x: model.coef_ * x + model.intercept_

y_pred = model.predict(np.arange(400).reshape(-1, 1))

score = model.score(x, y)
print(score)

plt.scatter(x, y)
plt.plot(range(400), y_pred, color="red")
# plt.plot(x, f(x), color="green")
plt.show()












































































