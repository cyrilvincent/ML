import pandas as pd
import sklearn.linear_model as lm
import numpy as np
import matplotlib.pyplot as plt

dataframe = pd.read_csv("data/house/house.csv")
dataframe = dataframe[dataframe.surface < 200]

a1 = np.array([1,2,3,4,5,6])
print(a1.shape)
mat1 = a1.reshape(-1, 3)
print(mat1)


x = dataframe.surface.values.reshape(-1, 1)
y = dataframe.loyer

# Instanciation du model
model = lm.LinearRegression()
# Apprentissage
model.fit(x, y)
print(model.coef_, model.intercept_)
# Prediction
predicted = model.predict(np.arange(200).reshape(-1, 1))
# Score
score = model.score(x, y)
print(score)

f = lambda x: model.coef_ * x + model.intercept_

plt.scatter(dataframe.surface, dataframe.loyer)
plt.plot(np.arange(200), predicted, color="red")
plt.plot(np.arange(200), f(np.arange(200)), color="yellow")
plt.show()
