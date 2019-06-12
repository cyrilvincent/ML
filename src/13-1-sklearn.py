import pandas as pd
import sklearn.linear_model as sklm

data = pd.read_csv('house/house.csv')
regr = sklm.LinearRegression()
#X = [[s] for s in data["surface"]]
X = data["surface"].values.reshape(-1,1)
print(data)
print(data["surface"])
print(X)
y = data["loyer"]
regr.fit(X,y)
print(regr.predict(X))
import math
print(regr.score(X, y))
print(regr.coef_)
print(regr.intercept_)

import matplotlib.pyplot as plt
plt.plot(data["surface"], data["loyer"], 'ro', markersize=4)
plt.plot(data["surface"], regr.predict(X) )
plt.show()






