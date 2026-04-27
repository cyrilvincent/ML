import pandas as pd
# pip install matplotlib
import matplotlib.pyplot as plt
import sklearn.linear_model as lm

df = pd.read_csv("data/house/house.csv")
print(df.describe())
df.hist()
plt.show()

plt.scatter(df["surface"], df["loyer"])


x = df["surface"].values.reshape(-1, 1)
y = df["loyer"]

# Créer le modèle
model = lm.LinearRegression()
# Apprentissage supervisé
model.fit(x, y)

print(model.coef_, model.intercept_)

print(model.score(x, y))

# Prédiction
ypred = model.predict(x)
plt.plot(x, ypred, color="red")
plt.show()
