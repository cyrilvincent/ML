import sklearn
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import sklearn.pipeline as pipe
import sklearn.preprocessing as pp
import sklearn.model_selection as ms
import numpy as np

print("Hello")
print(sklearn.__version__)

np.random.seed(42)

df = pd.read_csv("data/house/house.csv")
print(df.describe())

y = df["loyer"]
x = df["surface"].values.reshape(-1, 1)

xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, train_size=0.8, test_size=0.2)

# model = lm.LinearRegression()
model = pipe.make_pipeline(pp.PolynomialFeatures(2), lm.Ridge())
model.fit(xtrain, ytrain)
print("Train score ", model.score(xtrain, ytrain))
print("Test score: ", model.score(xtest, ytest))

x = np.arange(400).reshape(-1, 1)
ypred = model.predict(x)

plt.scatter(df["surface"], df["loyer"])
# plt.plot(x, model.coef_ * x + model.intercept_, color="red")
plt.plot(x, ypred, color="red")
plt.show()


