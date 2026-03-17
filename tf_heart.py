import tensorflow as tf
import tensorflow.keras as keras
import pandas as pd
import numpy as np
import sklearn.model_selection as ms

print(tf.__version__)

df = pd.read_csv("data/heartdisease/data_cleaned_up.csv")
print(df.describe())

np.random.seed(42)
tf.random.set_seed(42)

y = df["num"]
x = df.drop(["num"], axis=1)

xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, train_size=0.8, test_size=0.2)

model = keras.Sequential()
model.add(keras.layers.Input((x.shape[1],)))
model.add(keras.layers.Dense(5, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))

model.compile(optimizer="rmsprop", metrics=["accuracy"], loss="mse")
model.fit(xtrain, ytrain, epochs=10)
print(model.evaluate(xtest, ytest))

data = np.array([[28,1,2,130,132,0,2,185,0,0]])
ypred = model.predict(data)
print(ypred > 0.5)







