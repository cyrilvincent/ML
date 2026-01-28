import sklearn.linear_model as lm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.neural_network as nn
import sklearn.model_selection as ms
import numpy as np

np.random.seed(42)
dataframe = pd.read_csv("data/house/house.csv")

x = dataframe["surface"].values.reshape(-1, 1)
y = dataframe["loyer"]

xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, train_size=0.8, test_size=0.2,)

model = lm.LinearRegression()
model.fit(xtrain, ytrain)

xnew = np.arange(400).reshape(-1, 1)
ypredicted = model.predict(xnew)

score_train = model.score(xtrain, ytrain)
score_test = model.score(xtest, ytest)
print(f"Score: {score_train} {score_test}")

plt.scatter(dataframe["surface"], dataframe["loyer"])
plt.plot(np.arange(400), ypredicted, color="red")
plt.show()