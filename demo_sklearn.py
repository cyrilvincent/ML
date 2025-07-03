import sklearn.linear_model as lm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn.preprocessing as pp
import sklearn.pipeline as pipe

dataframe = pd.read_csv("data/house/house.csv")
dataframe=dataframe[dataframe.surface < 180]
y = dataframe["loyer"]
x = dataframe["surface"].values.reshape(-1, 1)
print(dataframe.shape)

# model = lm.LinearRegression()
model = pipe.make_pipeline(pp.PolynomialFeatures(500), lm.Ridge())
model.fit(x, y)

xtest = np.arange(180).reshape(-1, 1)
ypredicted = model.predict(xtest)
score = model.score(x, y)
print(f"score: {score}")

plt.scatter(dataframe["surface"], dataframe["loyer"])
plt.plot(xtest, ypredicted, color="red")
plt.show()

