import pandas as pd
import numpy as np
import sklearn
import sklearn.linear_model as lm
import matplotlib.pyplot as plt

print(sklearn.__version__)
df = pd.read_csv("data/house/house.csv")
print(df)

print(np.median(df.loyer / df.surface))

model = lm.LinearRegression()
model.fit(df.surface.values.reshape(-1, 1), df.loyer)
print(model.coef_, model.intercept_)

y_pred = model.predict(df.surface.values.reshape(-1, 1))
print(model.score(df.surface.values.reshape(-1, 1), df.loyer))

plt.scatter(df.surface, df.loyer)
plt.plot(df.surface.values, y_pred, color="red")
plt.show()




