import pandas as pd
import numpy as np
import sklearn.model_selection as ms
import sklearn.neighbors as nn
import sklearn.ensemble as rf
import sklearn.svm as svm
import matplotlib.pyplot as plt
import pickle
import sklearn.preprocessing as pp
import sklearn.neural_network as neural
import sklearn.metrics as metrics
import tensorflow.keras as keras


dataframe = pd.read_csv("data/breast-cancer/data.csv", index_col='id')
print(dataframe)
y = dataframe.diagnosis
x = dataframe.drop("diagnosis", 1)

# Fit
# Score = accuracy = nbtrue / len
# Predict KNN

np.random.seed(0)
xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, train_size=0.8, test_size=0.2)

scaler = pp.RobustScaler()
scaler.fit(xtrain)
xtrain = scaler.transform(xtrain)
xtest = scaler.transform(xtest)

model = keras.Sequential()
model.add(keras.layers.Dense(30, activation="relu", input_shape=(xtrain.shape[1],)))
model.add(keras.layers.Dense(10, activation="relu"))
model.add(keras.layers.Dense(1, activation="relu"))
model.compile(loss="mse", metrics="accuracy")
model.fit(xtrain, ytrain, epochs=10, batch_size=1)
score = model.evaluate(xtest, ytest)
print(score)
