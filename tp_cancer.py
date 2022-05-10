import pandas as pd
import sklearn.model_selection as ms
import sklearn.neighbors as ne
import numpy as np

np.random.seed(0)
dataframe = pd.read_csv("data/breast-cancer/data.csv")
y = dataframe.diagnosis
x = dataframe.drop(["diagnosis", "id"], axis=1)

xtrain, xtest, ytrain, ytest = ms.train_test_split(x,y, train_size=0.8, test_size=0.2)
model = ne.KNeighborsClassifier(n_neighbors=5)
model.fit(xtrain, ytrain)
score = model.score(xtest, ytest)
print(score)
