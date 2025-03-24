import sklearn
import sklearn.linear_model as lm
import pandas as pd
import matplotlib.pyplot as plt

print(sklearn.__version__)

dataframe = pd.read_csv("data/house/house.csv")

# Step 1 : Load data
plt.scatter(dataframe.surface, dataframe.loyer)

x = dataframe.surface.values.reshape(-1,1)
y = dataframe.loyer

# Step 2 : Instanciation du modèle : choix et création du modèle
model = lm.LinearRegression()

# Step 3 : Apprentissage
model.fit(x, y)

# Step 4 : Score
score = model.score(x, y) # (ypredicted - y) ** 2 avec un min_max_scaler à 0..1
print(f"Score: {score:.2f}")

# Step 5 : Predict
ypredicted = model.predict(x)

plt.plot(dataframe.surface, ypredicted, color="red")
plt.show()

