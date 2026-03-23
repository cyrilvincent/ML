import pandas as pd
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
import sklearn.pipeline as pipe
import sklearn.preprocessing as pre
import sklearn.model_selection as ms
import sklearn.neighbors as n
import sklearn.preprocessing as pp
import numpy as np


df = pd.read_csv("data/heartdisease/data_cleaned_up.csv")

print(df.describe())

y = df["num"]
x = df.drop(["num"], axis=1)

np.random.seed(42)

xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, train_size=0.8, test_size=0.2)

scaler = pp.RobustScaler()
scaler.fit(xtrain)
xtrain = scaler.transform(xtrain)
xtest = scaler.transform(xtest)

model = n.KNeighborsClassifier(n_neighbors=3)
model.fit(xtrain, ytrain)

trainscore = model.score(xtrain, ytrain)
testscore = model.score(xtest, ytest)
print(trainscore, testscore)




