import pandas as pd
import sklearn.linear_model as lm
import sklearn.neighbors as nn
import sklearn.model_selection as ms
import numpy as np
import sklearn.ensemble as rf
import matplotlib.pyplot as plt
import pickle

np.random.seed(0)
dataframe = pd.read_csv("data/breast-cancer/data.csv", index_col="id")
y = dataframe.diagnosis
x = dataframe.drop("diagnosis", axis=1)
xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, train_size=0.8, test_size=0.2)

with open("data/breast-cancer/rf.pickle", "rb") as f:
    model = pickle.load(f)

    y_predicted = model.predict(xtest)
    print(y_predicted)

