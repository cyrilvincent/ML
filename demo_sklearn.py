import sklearn.linear_model as lm
import sklearn
import sklearn.preprocessing as pp
import sklearn.pipeline as pipe
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

print(sklearn.__version__)

dataframe = pd.read_csv("data/house/house.csv")

x = dataframe.surface.values.reshape(-1, 1)
y = dataframe.loyer

model = lm.LinearRegression()
model.fit(x, y)
print(model.coef_, model.intercept_)
print(model.score(x, y))

f = lambda x: model.coef_ * x + model.intercept_

xpredict = np.arange(500).reshape(-1, 1)
ypredict = model.predict(xpredict)
ylambda = f(xpredict)
plt.scatter(x, y)
plt.plot(ypredict, color="red")
plt.plot(ylambda, color="green")
plt.show()