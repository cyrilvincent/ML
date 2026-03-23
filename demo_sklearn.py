import pandas as pd
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
import sklearn.pipeline as pipe
import sklearn.preprocessing as pre
import sklearn.model_selection as ms
import numpy as np

df = pd.read_csv("data/house/house.csv")

y = df["loyer"]
x = df["surface"].values.reshape(-1, 1)

np.random.seed(42)

xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, train_size=0.8, test_size=0.2)

# Instanciation du moddèle
# model = lm.LinearRegression()
model = pipe.make_pipeline(pre.PolynomialFeatures(2), lm.Ridge())

model.fit(xtrain, ytrain)

ypred = model.predict(np.arange(400).reshape(-1, 1))

scoretrain = model.score(xtrain, ytrain)
scoretest = model.score(xtest, ytest)
print(scoretrain)
print(scoretest)

plt.scatter(df["surface"], df["loyer"])
plt.plot(np.arange(400), ypred, color="red")
plt.show()
