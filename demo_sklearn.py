import sklearn
import pandas as pd
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
import numpy as np

print(sklearn.__version__)

#1 Load data
dataframe = pd.read_csv("data/house/house.csv")

#2 Make dataset
y = dataframe["loyer"]
x = dataframe["surface"].values.reshape(-1, 1)

#3 & 4

#5 Creating the model
model = lm.LinearRegression()
# f(x) = ax + b => 2 poids

#6 Fit
model.fit(x, y)

#7 Scoring (facultatif)
score = model.score(x, y)
print(f"Score: {score:.2f}")

#8 Predict
ypredicted = model.predict(x)

#9 Dataviz
plt.scatter(x, y)
xnew = np.arange(400)
ypredicted = model.predict(xnew.reshape(-1, 1))
plt.plot(xnew, ypredicted, color="red")
plt.show()
