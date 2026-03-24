import pandas as pd
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
import sklearn.pipeline as pipe
import sklearn.preprocessing as pre
import sklearn.model_selection as ms
import sklearn.neighbors as n
import sklearn.preprocessing as pp
import numpy as np
import sklearn.ensemble as rf
import sklearn.tree as tree
import sklearn.svm as svm
import sklearn.neural_network as neural
import tensorflow as tf
import keras
import pickle

df = pd.read_csv("data/breast-cancer/data.csv")

df["rnd"] = np.random.rand(len(df))
y = df["diagnosis"]
x = df.drop(["diagnosis", "id"], axis=1)

print(df.corr())

columns = x.columns

np.random.seed(42)
tf.random.set_seed(42)

xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, train_size=0.8, test_size=0.2)

scaler = pp.RobustScaler()
scaler.fit(xtrain)
xtrain = scaler.transform(xtrain)
xtest = scaler.transform(xtest)

model = keras.Sequential()
model.add(keras.layers.Input((xtrain.shape[1],)))
model.add(keras.layers.Dense(20, activation="relu"))
model.add(keras.layers.Dense(20, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))

model.compile(optimizer="rmsprop", metrics=["accuracy"], loss="mse")

# model.fit(xtrain, ytrain, validation_split=0.2)
model.fit(xtrain, ytrain, validation_data=(xtest, ytest), epochs=10)

ypred = model.predict(xtest)

print(model.evaluate(xtest, ytest))
