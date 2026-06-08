import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import sklearn.preprocessing as pp
import sklearn.pipeline as pipe
import sklearn.model_selection as ms
import numpy as np

print("Hello")
print(pd.__version__)

df = pd.read_csv("data/house/house.csv")
print(df)
print(df.describe())
print(df.corr())

y = df["loyer"]
x = df["surface"].values.reshape(-1, 1)

np.random.seed(42)

xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, train_size=0.8, test_size=0.2)

# model = lm.LinearRegression()
model = pipe.make_pipeline(pp.PolynomialFeatures(2), lm.Ridge())
model.fit(xtrain, ytrain)

train_score = model.score(xtrain, ytrain)
test_score = model.score(xtest, ytest)
print(train_score, test_score)

xnew = np.arange(400).reshape(-1, 1)
ypred = model.predict(xnew)


plt.scatter(df["surface"], df["loyer"])
plt.plot(np.arange(400), ypred, color="red")
plt.show()