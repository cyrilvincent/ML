# Avec pandas load data/breast-cancer/data.Csv
# y = diagnosis
# x tout sauf diagnosis et id
import sklearn
import pandas as pd
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
import numpy as np
import sklearn.neighbors as n
import sklearn.model_selection as ms

np.random.seed(42)

#1 Load data
dataframe = pd.read_csv("data/breast-cancer/data.csv")

#2 Make dataset
y = dataframe["diagnosis"]
x = dataframe.drop(["diagnosis", "id"], axis=1)

xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, train_size=0.8, test_size=0.2)

#5 Creating the model
# model = lm.LinearRegression()
model = n.KNeighborsClassifier(n_neighbors=11)
# f(x) = ax + b => 2 poids

#6 Fit
model.fit(xtrain, ytrain)

#7 Scoring (facultatif)
score = model.score(xtest, ytest)
print(f"Score: {score:.2f}")