import pandas as pd
import numpy as np
import sklearn.model_selection as ms
import sklearn.neighbors as nn
import sklearn.ensemble as rf
import sklearn.neural_network as neural
import sklearn.svm as svm
import matplotlib.pyplot as plt
import pickle

dataframe = pd.read_csv("data/breast-cancer/data.csv", index_col='id')
print(dataframe)
y = dataframe.diagnosis
x = dataframe.drop("diagnosis", 1)

# Fit
# Score = accuracy = nbtrue / len
# Predict KNN

np.random.seed(0)
xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, train_size=0.8, test_size=0.2)
model = neural.MLPClassifier(hidden_layer_sizes=(30,20,10), warm_start=True, alpha=0.01)
model.fit(xtrain, ytrain)
score = model.score(xtest, ytest)

with open(f"data/breast-cancer/mlp30_20_10_model_{score:.3f}.pickle", "wb") as f:
    pickle.dump(model, f)

print(score)


