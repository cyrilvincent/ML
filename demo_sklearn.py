import sklearn.linear_model as lm
import sklearn
import sklearn.preprocessing as pp
import sklearn.pipeline as pipe
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn.model_selection as ms


print(sklearn.__version__)

np.random.seed(0)

dataframe = pd.read_csv("data/house/house.csv")
dataframe = dataframe[dataframe.surface < 200]

x = dataframe.surface.values.reshape(-1, 1)
y = dataframe.loyer

xtrain, xtest, ytrain, ytest = ms.train_test_split(x,y,train_size=0.8,test_size=0.2)

model = lm.LinearRegression()
for i in range(5):
    model = pipe.make_pipeline(pp.PolynomialFeatures(i), lm.Ridge(0))
    model.fit(xtrain, ytrain)
    # print(model.coef_, model.intercept_)
    print(i, model.score(xtrain, ytrain))
    print(i, model.score(xtest, ytest))


# f = lambda x: model.coef_ * x + model.intercept_

xpredict = np.arange(200).reshape(-1, 1)
ypredict = model.predict(xpredict)
# ylambda = f(xpredict)
plt.scatter(x, y)
plt.plot(ypredict, color="red")
# plt.plot(ylambda, color="green")
plt.show()
