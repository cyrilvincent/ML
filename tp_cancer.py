import pandas as pd
import numpy as np
import sklearn.model_selection as ms
import sklearn.neighbors as nn

dataframe = pd.read_csv("data/breast-cancer/data.csv", index_col='id')
print(dataframe)
y = dataframe.diagnosis
x = dataframe.drop("diagnosis", 1)

# Fit
# Score = accuracy = nbtrue / len
# Predict KNN

np.random.seed(0)
xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, train_size=0.8, test_size=0.2)
model = nn.KNeighborsClassifier(n_neighbors=7)
model.fit(xtrain, ytrain)
print(model.score(xtest, ytest))

ypredict = model.predict(xtest)
errors = ypredict - ytest
print(errors.values)
score = 1 - len(errors[np.abs(errors) == 1]) / len(ytest)
print(score)
fp = 1 - len(errors[errors == 1]) / len(ytest)
print(fp)


