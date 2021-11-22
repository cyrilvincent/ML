import pandas as pd
import sklearn.linear_model as lm
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing as pp
import sklearn.pipeline as pipe
import sklearn.model_selection as ms
import numpy as np

np.random.seed(0)

dataframe = pd.read_csv("data/house/house.csv")
dataframe = dataframe[dataframe.surface < 200]

a1 = np.array([1,2,3,4,5,6])
print(a1.shape)
mat1 = a1.reshape(-1, 3)
print(mat1)


x = dataframe.surface.values.reshape(-1, 1)
y = dataframe.loyer

xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, train_size=0.8, test_size=0.2)


# Instanciation du model
# model = lm.LinearRegression()

model = pipe.make_pipeline(pp.PolynomialFeatures(3), lm.Ridge())
# Apprentissage
model.fit(xtrain, ytrain)
# print(model.coef_, model.intercept_)
# Prediction
predicted = model.predict(np.arange(200).reshape(-1, 1))
# Score
score = model.score(xtest, ytest)
print(score)
# score = model.score(xtrain, ytrain)
# print(score)

# f = lambda x: model.coef_ * x + model.intercept_

plt.scatter(dataframe.surface, dataframe.loyer)
plt.plot(np.arange(200), predicted, color="red")
# plt.plot(np.arange(200), f(np.arange(200)), color="yellow")
plt.show()
