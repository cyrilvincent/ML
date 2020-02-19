import tensorflow.keras as keras

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Set numeric type to float32 from uint8
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

# Normalize value to [0, 1]
x_train /= 255
x_test /= 255

# Reshape the dataset into 4D array
x_train = x_train.reshape(x_train.shape[0], 28*28)
x_test = x_test.reshape(x_test.shape[0], 28*28)

import tensorflow as tf
import tensorflow.keras as keras
model = keras.Sequential([
    keras.layers.Dense(500, activation=tf.nn.relu,
                       input_shape=(x_train.shape[1],)),
    keras.layers.Dense(200, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
  ])

model.compile(loss="categorical_crossentropy", optimizer="rmsprop",metrics=['accuracy'])
model.summary()

history = model.fit(x_train, y_train, epochs=200, batch_size=10, validation_split=0.2)
eval = model.evaluate(x_test, y_test)
print(eval)
