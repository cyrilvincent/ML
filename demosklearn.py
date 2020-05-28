import pandas as pd
import sklearn.linear_model as linearmodel

dataset = pd.read_csv("data/house/house.csv")
model = linearmodel.LinearRegression()

x = dataset.surface.values.reshape(-1,1) # shape(546) => (546,1) [1,2,3] x[1] => [[1,2,3]] x[0,1]
y = dataset.loyer
print(x)

import sklearn.model_selection as ms
xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, train_size=0.8, test_size=0.2)

model.fit(xtrain,ytrain)
print(model.score(xtest,ytest))
print(model.coef_, model.intercept_)
print(model.predict(xtest))

import matplotlib.pyplot as plt
import numpy as np
plt.scatter(x,y)
plt.plot(np.arange(400), model.predict(np.arange(400).reshape(-1,1)))
plt.show()