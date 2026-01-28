import sklearn.linear_model as lm
import pandas as pd
import numpy as np
import sklearn.neighbors as n
import sklearn.model_selection as ms
np.random.seed(42)

pd.set_option('display.max_columns', None)
dataframe = pd.read_csv("data/heartdisease/data_cleaned_up.csv")

print(dataframe.describe())

y = dataframe["num"]
x = dataframe.drop(["num"], axis=1)

xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, train_size=0.8, test_size=0.2,)

model = n.KNeighborsClassifier(n_neighbors=3)

model.fit(xtrain, ytrain)
score_train = model.score(xtrain, ytrain)
score_test = model.score(xtest, ytest)
print(score_train, score_test)