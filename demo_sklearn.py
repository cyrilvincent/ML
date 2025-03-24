import sklearn
import sklearn.linear_model as lm
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.preprocessing as pp
import sklearn.pipeline as pipe
import numpy as np

print(sklearn.__version__)

dataframe = pd.read_csv("data/house/house.csv")

# Step 1 : Load data
plt.scatter(dataframe.surface, dataframe.loyer)

x = dataframe.surface.values.reshape(-1,1)
y = dataframe.loyer

# Step 2 : Instanciation du modèle : choix et création du modèle
# model = lm.LinearRegression()
model = pipe.make_pipeline(pp.PolynomialFeatures(2), lm.Ridge())

# Step 3 : Apprentissage
model.fit(x, y)

# Step 4 : Score
score = model.score(x, y) # (ypredicted - y) ** 2 avec un min_max_scaler à 0..1
print(f"Score: {score:.2f}")

# Step 5 : Predict
xtest = np.arange(400)
ypredicted = model.predict(xtest.reshape(-1,1))

plt.plot(xtest, ypredicted, color="red")
plt.show()

