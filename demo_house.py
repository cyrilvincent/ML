import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import numpy as np
import sklearn.preprocessing as pp
import sklearn.pipeline as pipe
import sklearn.model_selection as ms

np.random.seed(0)

dataframe = pd.read_csv("data/house/house.csv")
dataframe = dataframe[dataframe.surface < 200]

print(dataframe.describe())

x = dataframe.surface
y = dataframe.loyer

xtrain, xtest, ytrain, ytest = ms.train_test_split(x,y,train_size=0.8, test_size=0.2)

model = lm.LinearRegression()
model = pipe.make_pipeline(pp.PolynomialFeatures(2), lm.Ridge())
model.fit(xtrain.values.reshape(-1, 1), ytrain)
print(model[0].get_feature_names_out())

# print(model.coef_, model.intercept_)

ypred = model.predict(np.arange(200).reshape(-1, 1))
rvalue = model.score(xtest.values.reshape(-1 ,1), ytest)
print(rvalue)
# print(np.mean((ypred - y) ** 2))

plt.scatter(x, y)
plt.plot(np.arange(200), ypred, color="red")
plt.show()

