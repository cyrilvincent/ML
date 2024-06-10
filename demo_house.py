import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import sklearn.pipeline as pipe
import sklearn.preprocessing as pp
import sklearn.model_selection as ms
import numpy as np

np.random.seed(42)

dataframe = pd.read_csv("data/house/house.csv")
dataframe = dataframe[dataframe.surface < 200]
y = dataframe.loyer
x = dataframe.surface.values.reshape(-1, 1)

xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, train_size=0.8, test_size=0.2)

# model = lm.LinearRegression()
model = pipe.make_pipeline(pp.PolynomialFeatures(2), lm.Ridge())
model.fit(xtrain, ytrain)
ypred = model.predict(xtest)
score = model.score(xtrain, ytrain)
print(score)
score = model.score(xtest, ytest)
print(score)

plt.scatter(dataframe.surface, dataframe.loyer)
plt.scatter(xtest, ypred, color="red")
plt.show()