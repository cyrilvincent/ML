import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import sklearn.model_selection as ms
import sklearn.preprocessing as pp

dataframe = pd.read_csv("data/rul/Battery_RUL.csv")
print(dataframe.describe())

plt.scatter(dataframe["Max. Voltage Dischar. (V)"], dataframe["RUL"])
plt.show()

y = dataframe["RUL"]
x = dataframe.drop(["RUL", "Cycle_Index"], axis=1)

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

model = tf.keras.Sequential([
    tf.keras.layers.Dense(20, activation="relu", input_shape=(xtrain.shape[1],)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(20, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(20, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(20, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])
model.compile(loss="mse")
model.fit(xtrain, ytrain, epochs=10, batch_size=1, validation_data=(xtest, ytest))
score = model.evaluate(xtest, ytest)
print(score, score ** 0.5, (score ** 0.5) * 550)

ypred = model.predict(xtest)
ypred = yscaler.inverse_transform(ypred)
print(ypred)





