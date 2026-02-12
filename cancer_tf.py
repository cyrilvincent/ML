import tensorflow as tf
import pandas as pd
import numpy as np
import sklearn.model_selection as ms
import sklearn.preprocessing as pp
import sklearn.neural_network as nn
import matplotlib.pyplot as plt

tf.random.set_seed(42)

dataframe = pd.read_csv("data/breast-cancer/data.csv")
y = dataframe["diagnosis"]
x = dataframe.drop(["diagnosis", "id"], axis=1)

xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, train_size=0.8, test_size=0.2)

scaler = pp.RobustScaler()
scaler.fit(xtrain)
xtrain = scaler.transform(xtrain)
xtest = scaler.transform(xtest)

model = tf.keras.Sequential()
print(x.shape)

model.add(tf.keras.layers.Dense(40, activation="relu", input_shape=(x.shape[1],)))
model.add(tf.keras.layers.Dense(25, activation="relu"))
model.add(tf.keras.layers.Dense(10, activation="relu"))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

model.compile(loss="mse", optimizer="rmsprop", metrics=["accuracy"])
model.summary()

hist = model.fit(xtrain, ytrain, epochs=50, validation_split=0.2)
score = model.evaluate(xtrain, xtrain)
print(score)
ypredicted = model.predict(xtest)

plt.plot(hist.history["accuracy"])
plt.show()

model.save("data/breast-cancer/mlp.h5")



