import tensorflow as tf
import tensorflow.keras as keras
import pandas as pd
import numpy as np
import sklearn.model_selection as ms
import sklearn.preprocessing as pp
import pickle

print(tf.__version__)

df = pd.read_csv("data/breast-cancer/data.csv")
print(df.describe())

np.random.seed(42)
tf.random.set_seed(42)

y = df["diagnosis"]
x = df.drop(["diagnosis", "id"], axis=1)

xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, train_size=0.8, test_size=0.2)

scaler = pp.RobustScaler()
scaler.fit(xtrain)
xtrain = scaler.transform(xtrain)
xtest = scaler.transform(xtest)

np.savetxt("data/breast-cancer/scaler_center.txt", scaler.center_)
np.savetxt("data/breast-cancer/scaler_scale.txt", scaler.scale_)

with open("data/breast-cancer/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# scaler.inverse_transform(xtest)

model = keras.Sequential()
model.add(keras.layers.Input((x.shape[1],)))
model.add(keras.layers.Dense(20, activation="relu"))
model.add(keras.layers.Dense(10, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))

model.compile(optimizer="rmsprop", metrics=["accuracy"], loss="mse")
model.fit(xtrain, ytrain, epochs=10, validation_split=0.2)
print(model.evaluate(xtest, ytest))
model.save("data/breast-cancer/mlp.keras")

data = np.array([[17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189]])
data = scaler.transform(data)
print(data)
ypred = model.predict(data)
print(ypred > 0.5)







