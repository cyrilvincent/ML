import tensorflow.keras as keras
import tensorflow as tf
import pandas


tf.random.set_seed(1)

dataframe = pandas.read_csv("data/breast-cancer/data.csv", index_col="id")
y = dataframe.diagnosis
x = dataframe.drop("diagnosis", 1)

import tensorflow as tf
import tensorflow.keras as keras
model = keras.Sequential([
    keras.layers.Dense(30, activation=tf.nn.relu,
                       input_shape=(x.shape[1],)),
    keras.layers.Dense(30, activation=tf.nn.relu),
    keras.layers.Dense(30, activation=tf.nn.relu),
    keras.layers.Dense(1)
  ])

model.compile(loss="mse", optimizer="rmsprop",metrics=['accuracy'])
model.summary()

history = model.fit(x, y, epochs=10, batch_size=1, validation_split=0.2)
eval = model.evaluate(x, y)
print(eval)
