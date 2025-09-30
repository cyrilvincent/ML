import sklearn
import pandas as pd
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
import numpy as np
import sklearn.neighbors as n
import sklearn.model_selection as ms
import sklearn.preprocessing as pp

print(sklearn.__version__)
np.random.seed(42)

#1 Load data
dataframe = pd.read_csv("data/heartdisease/data_cleaned_up.csv")

#2 Make dataset
y = dataframe["num"]
x = dataframe.drop("num", axis=1)

xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, train_size=0.8, test_size=0.2)

scaler = pp.RobustScaler()
scaler.fit(xtrain)
xtrain = scaler.transform(xtrain)
xtest = scaler.transform(xtest)
# scaler.inverse_transform(x)

#5 Creating the model
# model = lm.LinearRegression()
for k in range(3,12,2):
    model = n.KNeighborsClassifier(n_neighbors=k)
    # f(x) = ax + b => 2 poids

    #6 Fit
    model.fit(xtrain, ytrain)

    #7 Scoring (facultatif)
    score = model.score(xtest, ytest)
    print(f"Score: {k} {score:.2f}")
