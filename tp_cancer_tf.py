import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import sklearn.preprocessing as pp
import sklearn.pipeline as pipe
import sklearn.model_selection as ms
import numpy as np
import sklearn.neighbors as n
import sklearn.ensemble as rf
import sklearn.tree as tree
import pickle
import emlearn
import sklearn.svm as svm
import sklearn.neural_network as nn
import tensorflow
import keras

df = pd.read_csv("data/breast-cancer/data.csv")

y = df["diagnosis"]
x = df.drop(["diagnosis", "id"], axis=1)

np.random.seed(42)
keras.utils.set_random_seed(42)

xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, train_size=0.8, test_size=0.2)

scaler = pp.RobustScaler()
scaler.fit(xtrain)
xtrain = scaler.transform(xtrain)
xtest = scaler.transform(xtest)

model = keras.Sequential()
model.add(keras.layers.Input((xtrain.shape[1],)))
model.add(keras.layers.Dense(20, activation="relu"))
model.add(keras.layers.Dense(10, activation="relu"))
model.add(keras.layers.Dense(2, activation="softmax"))

model.compile(optimizer="rmsprop", metrics=["accuracy"], loss="mse")

model.fit(xtrain, ytrain, epochs=10, validation_split=0.2)  # validation_data=(xtest, ytest))

print(model.evaluate(xtest, ytest))

ypred = model.predict(xtest)
print(np.argmax(ypred))

model.save("data/breast-cancer/mlp.h5")
