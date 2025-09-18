import numpy as np
import sklearn
import sklearn.linear_model as lm
import pandas as pd
import matplotlib.pyplot as plt
print(sklearn.__version__)


# 1 Pandas DataMart
dataframe = pd.read_csv("data/house/house.csv")

# Déterminer x et y
y = dataframe["loyer"]
x = dataframe["surface"].values.reshape(-1, 1)

# 3 Model
model = lm.LinearRegression()
# f(x) = ax + b; a = slope; b = intersection pour x=0;
# a et b soit les monômes du polynôme de dégré 1; a et b sont des poids

# 4 Apprentissage supervisé car y est connu
model.fit(x, y)

# 5 Score
score = model.score(x, y)
print("Score", score)

# 6 Predict
xnew = np.arange(400).reshape(-1, 1)
ypredicted = model.predict(xnew)

# 7 Dataviz
plt.scatter(dataframe["surface"], dataframe["loyer"])
plt.plot(xnew, ypredicted, color="red")

plt.show()

