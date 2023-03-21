import matplotlib.pyplot as plt
import pandas as pd
import sklearn.linear_model as lm
import sklearn.pipeline as pipe
import sklearn.preprocessing as pp
import sklearn.model_selection as ms
import numpy as np

df = pd.read_csv("data/house/house.csv")
df = df[df.surface < 175]
print(df)


np.random.seed(0)
xtrain, xtest, ytrain, ytest = ms.train_test_split(df.surface.values.reshape(-1, 1), df.loyer)

# model = lm.LinearRegression()
model = pipe.make_pipeline(pp.PolynomialFeatures(2), lm.Ridge())

model.fit(xtrain.reshape(-1, 1), ytrain)
# /!\ JAMAIS fit sur testset

loyer_predict = model.predict(xtest.reshape(-1, 1))

print(model.score(xtest.reshape(-1, 1), ytest))
# print(model.coef_, model.intercept_)

plt.scatter(df.surface, df.loyer)
plt.scatter(xtest, loyer_predict, color="red")
plt.show()