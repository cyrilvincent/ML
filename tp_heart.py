# Charger le fichier heartdisease/data_cleaned_up.csv
# Describe
# ok = dataset où num == 0
# ko = dataset où num == 1
# par 2 simples describe sur ok et ko trouver des corrélations
# sur le dataframe initial tester la méthode .corr()

import pandas as pd
import sklearn.linear_model as lm
import sklearn.model_selection as ms
import numpy as np

dataframe = pd.read_csv("data/heartdisease/data_cleaned_up.csv")
ok = dataframe[dataframe.num == 0]
ko = dataframe[dataframe.num == 1]

print(ok.describe().T)
print(ko.describe().T)

print(dataframe.corr())

y = dataframe.num
x = dataframe.drop("num", axis=1)

np.random.seed(0)

xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, train_size=0.8, test_size=0.2)

model = lm.LinearRegression()
model.fit(xtrain, ytrain)

print(model.score(xtest, ytest))
print(model.score(xtrain, ytrain))

