import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import sklearn.model_selection as ms
import sklearn.preprocessing as pp

dataframe = pd.read_csv("data/rul/Battery_RUL.csv")
print(dataframe.describe())

plt.subplot(211)
plt.scatter(dataframe["Max. Voltage Dischar. (V)"], dataframe["RUL"])
plt.subplot(212)
plt.scatter(dataframe["Cycle_Index"], dataframe["RUL"], color="red")
plt.show()

y = dataframe["RUL"]
x = dataframe.drop("RUL", axis=1)

np.random.seed(42)
tf.random.set_seed(42)

xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y)

scaler = pp.RobustScaler()
scaler.fit(xtrain)
xtrain = scaler.transform(xtrain)
xtest = scaler.transform(xtest)

ytrain = ytrain.values.reshape(-1, 1)
ytest = ytest.values.reshape(-1, 1)
yscaler = pp.MinMaxScaler()
yscaler.fit(ytrain)
ytrain = yscaler.transform(ytrain)
ytest = yscaler.transform(ytest)

xtrain = xtrain.reshape(xtrain.shape[0], 1, xtrain.shape[1])
xtest = xtest.reshape(xtest.shape[0], 1, xtest.shape[1])

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, input_shape=(xtrain.shape[1],xtrain.shape[2])),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])
model.compile(loss="mse")
model.fit(xtrain, ytrain, epochs=10, batch_size=1, validation_data=(xtest, ytest))
score = model.evaluate(xtest, ytest)
print(score, score ** 0.5, (score ** 0.5) * 550)

ypred = model.predict(xtest)
ypred = yscaler.inverse_transform(ypred)
print(ypred)





