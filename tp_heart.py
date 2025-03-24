import pandas as pd
import sklearn.linear_model as lm
import numpy as np
import sklearn.neighbors as n
import sklearn.model_selection as ms



dataframe = pd.read_csv("data/heartdisease/data_cleaned_up.csv")

y = dataframe.num
x = dataframe.drop("num", axis=1)

xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, test_size=0.2, train_size=0.8)

# model = lm.LinearRegression()
model = n.KNeighborsClassifier(n_neighbors=3)

model.fit(xtrain, ytrain)

score = model.score(xtrain, ytrain)
print(score)
score = model.score(xtest, ytest)
print(score)

predicted = model.predict(xtest)
print(predicted)

# LinearModel
# Fit
# Predict
# Score
