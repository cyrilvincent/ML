import pipes

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn.linear_model as lm
import sklearn.pipeline as pipe
import sklearn.preprocessing as pp
import matplotlib.pyplot as plt
import sklearn.model_selection as ms

# Pandas
dataframe = pd.read_csv("data/house/house.csv")


dataframe = dataframe[dataframe.surface < 200]

dataframe["loyer_per_m2"] = dataframe.loyer / dataframe.surface
print(dataframe.head())
print(dataframe.describe())



plt.scatter(dataframe.surface, dataframe.loyer)
# plt.show()

# Instanciation du model
# model = lm.LinearRegression()
model = pipe.make_pipeline(pp.PolynomialFeatures(3), lm.Ridge())
# Apprentissage - Fit
model.fit(dataframe.surface.values.reshape(-1,1), dataframe.loyer)
# Prediction
predicted = model.predict(np.arange(200).reshape(-1,1))
print(predicted)
# Scoring
print(model.score(dataframe.surface.values.reshape(-1,1), dataframe.loyer)) # = np.mean((predicted - dataframe.loyer) ** 2)

plt.scatter(dataframe.surface, dataframe.loyer)
plt.plot(np.arange(200), predicted, color="red")
plt.show()
