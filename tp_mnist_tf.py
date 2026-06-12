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
import tensorflow
import keras

data = np.load("data/mnist/mnist.npz")
xtrain = data["x_train"] #[::10]
xtest = data["x_test"]
ytrain = data["y_train"]
ytest = data["y_test"]

print(xtrain.shape)

xtrain = xtrain.reshape(-1, 28*28)
xtest = xtest.reshape(-1, 28*28)

# ytrain = keras.utils.to_categorical(ytrain)
# ytest = keras.utils.to_categorical(ytest)

model = keras.Sequential()
model.add(keras.layers.Input((xtrain.shape[1],)))
model.add(keras.layers.Dense(500, activation="relu"))
model.add(keras.layers.Dense(200, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))

model.summary()

model.compile(optimizer="rmsprop", metrics=["accuracy"], loss="mse")

model.fit(xtrain, ytrain, epochs=1, validation_data=(xtest, ytest))

print(model.evaluate(xtest, ytest))

ypred = model.predict(xtest)
pred = np.argmax(ypred, axis=1)
print(pred)
