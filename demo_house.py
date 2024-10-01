import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import sklearn.linear_model as lm


dataframe = pd.read_csv("data/house/house.csv")
dataframe = dataframe[dataframe["surface"]<200]
print(dataframe.describe())

slope, intercept, rvalue, pvalue, mse = stats.linregress(dataframe["surface"], dataframe["loyer"])
print(slope, intercept, rvalue, pvalue, mse)

x = np.arange(200)
y = slope * x + intercept
plt.plot(x, y, color="red")


x = dataframe["surface"].values.reshape(-1,1)
y = dataframe["loyer"]

model = lm.LinearRegression()
model.fit(x, y)
ypred = model.predict(x)
print(model.coef_, model.intercept_) # Variables privées non portables entre modèles
score = model.score(x,y)
print(score)
plt.plot(dataframe["surface"], ypred, color="green")

plt.scatter(dataframe["surface"], dataframe["loyer"])
plt.show()
