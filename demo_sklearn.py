import matplotlib.pyplot as plt
import pandas as pd
import sklearn.linear_model as lm

df = pd.read_csv("data/house/house.csv")
df = df[df.surface < 175]
print(df)

model = lm.LinearRegression()
model.fit(df.surface.values.reshape(-1, 1), df.loyer)

loyer_predict = model.predict(df.surface.values.reshape(-1, 1))

print(model.score(df.surface.values.reshape(-1, 1), df.loyer))
print(model.coef_, model.intercept_)

plt.scatter(df.surface, df.loyer)
plt.plot(df.surface, loyer_predict, color="red")
plt.show()