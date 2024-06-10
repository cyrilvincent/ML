import pandas as pd
import sklearn.neighbors as neighbors
import sklearn.model_selection as ms
import numpy as np

dataframe = pd.read_csv("data/breast-cancer/data.csv", index_col="id")
print(dataframe)
y = dataframe.diagnosis
x = dataframe.drop("diagnosis", axis=1)
np.random.seed(42)

xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, train_size=0.8, test_size=0.2)
print(xtest.shape, ytest.shape)

for k in range(1, 15, 2):
    model = neighbors.KNeighborsClassifier(n_neighbors=k)
    model.fit(xtrain, ytrain)
    print(model.score(xtrain, ytrain), model.score(xtest, ytest))

ypred = model.predict(xtest)
print(ypred)
