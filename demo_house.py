import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import sklearn.preprocessing as pp
import sklearn.pipeline as pipe
import sklearn.model_selection as ms

dataframe = pd.read_csv("data/house/house.csv")
dataframe = dataframe[dataframe.surface < 180]
plt.scatter(dataframe.surface, dataframe.loyer)

x = dataframe.surface.values.reshape(-1, 1)
y = dataframe.loyer

np.random.seed(42)
xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, train_size=0.8, test_size=0.2)

# model = lm.LinearRegression()
model = pipe.make_pipeline(pp.PolynomialFeatures(2), lm.Ridge())


model.fit(xtrain, ytrain)
# print(model.coef_, model.intercept_)
xpredicted = np.arange(180).reshape(-1, 1)
ypredicted = model.predict(xtest)

plt.plot(xtest, ypredicted, color="red")
print(model.score(xtest, ytest))
print(model.score(xtrain, ytrain))
plt.show()
