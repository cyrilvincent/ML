import pandas as pd
import sklearn.linear_model as sklm

data = pd.read_csv('data/house/house.csv')
model = sklm.LinearRegression()
x = data["surface"].values.reshape(-1,1)
y = data["loyer"]

# Effectuer une regression lineaire + afficher + score s'inspirer de demosklearn
# Effectuer une regression polynomiale + afficher + score s'inspirer de demosklearn + 13-4-sklearn-polynomial

import matplotlib.pyplot as plt
plt.scatter(x, y)
# import sklearn.model_selection as ms
# xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, train_size=0.8, test_size=0.2)

import sklearn.preprocessing as pp
import sklearn.pipeline as pipe
import numpy as np
model = pipe.make_pipeline(pp.PolynomialFeatures(1), sklm.Ridge())
model.fit(x,y)
ypred = model.predict(np.arange(400).reshape(-1,1))
print(model.score(x,y))
plt.plot(np.arange(400).reshape(-1,1),ypred)
model = pipe.make_pipeline(pp.PolynomialFeatures(6), sklm.Ridge())
model.fit(x,y)
ypred = model.predict(np.arange(400).reshape(-1,1))
print(model.score(x,y))
plt.plot(np.arange(400).reshape(-1,1),ypred)

plt.show()