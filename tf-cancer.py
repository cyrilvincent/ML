import tensorflow as tf
import pandas
import numpy as np
import sklearn.model_selection as ms
import sklearn.preprocessing as pp


tf.random.set_seed(42)

dataframe = pandas.read_csv("data/breast-cancer/data.csv", index_col="id")
y = dataframe.diagnosis
x = dataframe.drop(["diagnosis"], axis=1)

np.random.seed(42)
xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, train_size=0.8, test_size=0.2)

scaler = pp.RobustScaler()
scaler.fit(xtrain)
xtrain = scaler.transform(xtrain)
xtest = scaler.transform(xtest)

# ytrain = tf.keras.utils.to_categorical(ytrain)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(30, activation=tf.nn.relu, input_shape=(x.shape[1],)),
    tf.keras.layers.Dense(30, activation=tf.nn.relu),
    tf.keras.layers.Dense(30, activation=tf.nn.relu),
    tf.keras.layers.Dense(1, activation=tf.nn.softmax)
  ])

model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=['accuracy'])
model.summary()

hist = model.fit(xtrain, ytrain, epochs=50, batch_size=1, validation_data=(xtest, ytest))
eval = model.evaluate(xtest, ytest)
print(eval)
ypredict = model.predict(xtest)


import matplotlib.pyplot as plt
f, ax = plt.subplots()
ax.plot([None] + hist.history['accuracy'], 'o-')
ax.legend(['Train accuracy'], loc = 0)
ax.set_title('Accuracy per Epoch')
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
plt.show()

