import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm

dataframe = pd.read_csv("data/house/house.csv")
dataframe = dataframe[dataframe.surface < 180]
plt.scatter(dataframe.surface, dataframe.loyer)

x = dataframe.surface.values.reshape(-1, 1)
y = dataframe.loyer
model = lm.LinearRegression()
model.fit(x, y)
print(model.coef_, model.intercept_)
ypredicted = model.predict(x)
plt.plot(x, ypredicted, color="red")
print(model.score(x, y))

plt.show()
