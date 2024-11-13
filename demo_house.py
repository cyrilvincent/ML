import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as lm

dataframe = pd.read_csv("data/house/house.csv")

# 0 Choix des data
y = dataframe.loyer
x = dataframe.surface.values.reshape(-1, 1)


# 1 Instanciation du modèle
model = lm.LinearRegression()

# 2 Fit - Apprentissage
model.fit(x, y)

predicted = model.predict(x)

plt.scatter(dataframe.surface, dataframe.loyer)
plt.plot(dataframe.surface, predicted, color="red")
plt.show()
