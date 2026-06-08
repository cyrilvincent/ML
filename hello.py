import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import numpy as np

print("Hello")
print(pd.__version__)

df = pd.read_csv("data/house/house.csv")
print(df)
print(df.describe())
print(df.corr())

y = df["loyer"]
x = df["surface"].values.reshape(-1, 1)

model = lm.LinearRegression()
model.fit(x, y)

xnew = np.arange(400).reshape(-1, 1)
ypred = model.predict(xnew)


plt.scatter(df["surface"], df["loyer"])
plt.plot(np.arange(400), ypred, color="red")
plt.show()