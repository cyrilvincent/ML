import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import numpy as np
import sklearn.pipeline as pipe
import sklearn.preprocessing as pp

print("Hello")
print(pd.__version__)

df = pd.read_csv("data/house/house.csv")
print(df.describe())

plt.scatter(df["surface"], df["loyer"])

y = df["loyer"]
x = df["surface"].values.reshape(-1, 1)

# model = lm.LinearRegression()
model = pipe.make_pipeline(pp.PolynomialFeatures(2), lm.Ridge())
model.fit(x, y)
# print(model.coef_, model.intercept_)

print(model.score(x, y))


x = np.arange(400).reshape(-1, 1)
ypred = model.predict(x)
plt.plot(x, ypred, color="red")
plt.show()
