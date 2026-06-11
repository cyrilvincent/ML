import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import numpy as np
import sklearn.pipeline as pipe
import sklearn.preprocessing as pp
import sklearn.model_selection as ms

print("Hello")
print(pd.__version__)

df = pd.read_csv("data/house/house.csv")
print(df.describe())

plt.scatter(df["surface"], df["loyer"])

y = df["loyer"]
x = df["surface"].values.reshape(-1, 1)

xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, train_size=0.8, test_size=0.2)

# model = lm.LinearRegression()
model = pipe.make_pipeline(pp.PolynomialFeatures(2), lm.Ridge())
model.fit(xtrain, ytrain)
# print(model.coef_, model.intercept_)

print("Training Score", model.score(xtrain, ytrain))
print("Testing Score", model.score(xtest, ytest))

x = np.arange(400).reshape(-1, 1)
ypred = model.predict(x)
plt.plot(x, ypred, color="red")
plt.show()
