import sklearn.linear_model as lm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.neural_network as nn

dataframe = pd.read_csv("data/house/house.csv")

x = dataframe["surface"].values.reshape(-1, 1)
y = dataframe["loyer"]

model = lm.LinearRegression()
model.fit(x, y)

xnew = np.arange(400).reshape(-1, 1)
ypredicted = model.predict(xnew)
score = model.score(x, y)
print(f"Score: {score}")

plt.scatter(dataframe["surface"], dataframe["loyer"])
plt.plot(np.arange(400), ypredicted, color="red")
plt.show()