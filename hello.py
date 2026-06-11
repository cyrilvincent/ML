import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import numpy as np

print("Hello")
print(pd.__version__)

df = pd.read_csv("data/house/house.csv")
print(df.describe())

plt.scatter(df["surface"], df["loyer"])


y = df["loyer"]
x = df["surface"].values.reshape(-1, 1)

model = lm.LinearRegression()
model.fit(x, y)
print(model.coef_, model.intercept_)

x = np.arange(400).reshape(-1, 1)
ypred = model.predict(x)

plt.plot(x, ypred, color="red")
plt.show()
