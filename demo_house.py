import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import numpy as np
import sklearn.preprocessing as pp
import sklearn.pipeline as pipe

dataframe = pd.read_csv("data/house/house.csv")
dataframe = dataframe[dataframe.surface < 200]
print(dataframe.describe())

x = dataframe.surface
y = dataframe.loyer

model = lm.LinearRegression()
model = pipe.make_pipeline(pp.PolynomialFeatures(50), lm.Ridge())
model.fit(x.values.reshape(-1, 1), y)
# print(model.coef_, model.intercept_)

ypred = model.predict(np.arange(200).reshape(-1, 1))
rvalue = model.score(x.values.reshape(-1 ,1), y)
print(rvalue)
# print(np.mean((ypred - y) ** 2))

plt.scatter(x, y)
plt.plot(np.arange(200), ypred, color="red")
plt.show()

