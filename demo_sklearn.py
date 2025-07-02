import sklearn.linear_model as lm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataframe = pd.read_csv("data/house/house.csv")
y = dataframe["loyer"]
x = dataframe["surface"].values.reshape(-1, 1)
print(dataframe.shape)

model = lm.LinearRegression()
model.fit(x, y)

xtest = np.arange(400).reshape(-1, 1)
ypredicted = model.predict(xtest)
score = model.score(x, y)
print(f"score: {score}")

plt.scatter(dataframe["surface"], dataframe["loyer"])
plt.plot(xtest, ypredicted, color="red")
plt.show()

