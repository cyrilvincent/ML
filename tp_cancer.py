import pandas as pd
import numpy as np
import sklearn.model_selection as ms
import sklearn.preprocessing as pp
import sklearn.neural_network as nn

np.random.seed(42)

dataframe = pd.read_csv("data/breast-cancer/data.csv")
y = dataframe["diagnosis"]
x = dataframe.drop(["diagnosis", "id"], axis=1)

xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, train_size=0.8, test_size=0.2)

scaler = pp.RobustScaler()
scaler.fit(xtrain)
xtrain = scaler.transform(xtrain)
xtest = scaler.transform(xtest)

model = nn.MLPClassifier(hidden_layer_sizes=(50,50,50), activation="relu")

model.fit(xtrain, ytrain)
score = model.score(xtest, ytest)

print(score)