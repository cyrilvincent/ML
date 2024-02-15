import pandas as pd
import sklearn.neighbors as nn
import sklearn.model_selection as ms
import numpy as np

dataframe = pd.read_csv("data/breast-cancer/data.csv", index_col="id")
print(dataframe.describe().T)

y = dataframe.diagnosis
x = dataframe.drop("diagnosis", axis=1)

np.random.seed(0)

xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, train_size=0.8, test_size=0.2)

model = nn.KNeighborsClassifier(n_neighbors=3)

model.fit(xtrain, ytrain)

print(model.score(xtest, ytest), model.score(xtrain, ytrain))