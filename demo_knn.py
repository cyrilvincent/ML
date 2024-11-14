import pandas as pd
import sklearn.neighbors as neighbors
import sklearn.model_selection as ms
import numpy as np

dataframe = pd.read_csv("data/heartdisease/data_cleaned_up.csv")
y = dataframe.num
x = dataframe.drop("num", axis=1)

np.random.seed(42)
xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, train_size=0.8, test_size=0.2)

model = neighbors.KNeighborsClassifier(n_neighbors=3)
model.fit(xtrain, ytrain)

score = model.score(xtest, ytest)
print(score)


