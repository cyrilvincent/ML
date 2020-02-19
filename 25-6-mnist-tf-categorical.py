import tensorflow.keras as keras
import tensorflow.compat.v1 as tf

import numpy as np

with np.load("data/mnist/mnist.npz", allow_pickle=True) as f:
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']

sample = np.random.randint(60000, size=5000)
data = x_train[sample]
target = y_train[sample]

# Set numeric type to float32 from uint8
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

model = keras.Sequential([
    keras.layers.Dense(784, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(0.0001)),
    keras.layers.Dense(500, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(0.0001)),
    keras.layers.Dense(500, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(0.0001)),
    keras.layers.Dense(500, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(0.0001)),
    keras.layers.Dense(500, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(0.0001)),
    keras.layers.Dense(500, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(0.0001)),
    keras.layers.Dense(10, activation=tf.nn.softmax),
  ])

model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
trained = model.fit(x_train, y_train, epochs=10, batch_size=120, validation_data=(x_test, y_test))
print(model.summary())

