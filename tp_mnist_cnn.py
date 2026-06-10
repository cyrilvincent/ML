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

xtrain = xtrain.astype(np.float32)
xtest = xtest.astype(np.float32)
xtrain /= 255
xtest /= 255

ytrain = keras.utils.to_categorical(ytrain)
ytest = keras.utils.to_categorical(ytest)

model = keras.Sequential()
model.add(keras.layers.Input((28, 28, 1)))
# CNN = Bottleneck
model.add(keras.layers.Conv2D(4, (3, 3), padding="same")) # 28,28,4
model.add(keras.layers.ReLU())
model.add(keras.layers.MaxPooling2D((2,2))) # 14,14,4

model.add(keras.layers.Conv2D(8, (3, 3), padding="same")) # 14,14,8
model.add(keras.layers.ReLU())
model.add(keras.layers.MaxPooling2D((2,2))) # 7,7,8

model.add(keras.layers.Flatten())  # 7 * 7 * 8 = 392

model.add(keras.layers.Dense(200, activation="relu"))
model.add(keras.layers.Dropout(0.1))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dropout(0.1))
model.add(keras.layers.Dense(10, activation="softmax"))

model.summary()

model.compile(optimizer="rmsprop", metrics=["accuracy"], loss="categorical_crossentropy")

model.fit(xtrain, ytrain, epochs=10, validation_data=(xtest, ytest), batch_size=10)

print(model.evaluate(xtest, ytest))

ypred = model.predict(xtest)
pred = np.argmax(ypred, axis=1)
print(pred)
