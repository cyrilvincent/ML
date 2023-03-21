import pandas as pd
import numpy as np
import sklearn.model_selection as ms
import sklearn.neighbors as nn

df = pd.read_csv("data/breast-cancer/data.csv", index_col="id")
np.random.seed(0)
y = df.diagnosis
x = df.drop("diagnosis", 1)

xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, train_size=0.8, test_size=0.2)
model = nn.KNeighborsClassifier(n_neighbors=5)
model.fit(xtrain, ytrain)
score = model.score(xtest, ytest)
print(score)
ypred = model.predict(xtest)
print((ypred - ytest).values)
