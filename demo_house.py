import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as lm

dataframe = pd.read_csv("data/house/house.csv")
dataframe = dataframe[dataframe.surface < 200]
y = dataframe.loyer
x = dataframe.surface.values.reshape(-1, 1)



model = lm.LinearRegression()
model.fit(x, y)
ypred = model.predict(x)
score = model.score(x, y)
print(score)


plt.scatter(dataframe.surface, dataframe.loyer)
plt.plot(x, ypred, color="red")
plt.show()