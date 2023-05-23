import pandas as pd
import numpy as np
import sklearn
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
import sklearn.preprocessing as pp
import sklearn.pipeline as pipe
import sklearn.model_selection as ms

def train(model):
    model.fit(df.surface.values.reshape(-1, 1), df.loyer)
    return model

print(sklearn.__version__)
df = pd.read_csv("data/house/house.csv")
print(df)




print(np.median(df.loyer / df.surface))

# model = lm.LinearRegression()
model = pipe.make_pipeline(pp.PolynomialFeatures(10), lm.Ridge())
model = train(model)


# model.fit(df.surface.values.reshape(-1, 1), df.loyer)
#print(model.coef_, model.intercept_)

y_pred = model.predict(np.arange(0, 400).reshape(-1, 1))
print(model.score(df.surface.values.reshape(-1, 1), df.loyer))

plt.scatter(df.surface, df.loyer)
plt.plot(np.arange(0, 400).reshape(-1, 1), y_pred, color="red")
plt.show()

def add(x, y):
    return x + y




