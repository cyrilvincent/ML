import pandas as pd
# pip install matplotlib
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import sklearn.preprocessing as pp
import sklearn.pipeline as pipe
import numpy as np
import sklearn.model_selection as ms


df = pd.read_csv("data/house/house.csv")
print(df.describe())
df.hist()
plt.show()

plt.scatter(df["surface"], df["loyer"])


x = df["surface"].values.reshape(-1, 1)
y = df["loyer"]

np.random.seed(42)

xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, train_size=0.8, test_size=0.2)

# Créer le modèle
# model = lm.LinearRegression()
model = pipe.make_pipeline(pp.PolynomialFeatures(2), lm.Ridge())
# Apprentissage supervisé
model.fit(xtrain, ytrain)

# print(model.coef_, model.intercept_)

print(model.score(xtrain, ytrain))
print(model.score(xtest, ytest))

x = np.arange(400)

# Prédiction
ypred = model.predict(x.reshape(-1, 1))
plt.plot(x, ypred, color="red")
plt.show()
