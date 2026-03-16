import sklearn
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as lm

print("Hello")
print(sklearn.__version__)

df = pd.read_csv("data/house/house.csv")
print(df.describe())

y = df["loyer"]
x = df["surface"].values.reshape(-1, 1)

model = lm.LinearRegression()
model.fit(x, y)

ypred = model.predict(x)

print(model.score(x, y))

plt.scatter(df["surface"], df["loyer"])
plt.plot(x, model.coef_ * x + model.intercept_, color="red")
plt.show()


