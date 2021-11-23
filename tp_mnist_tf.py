import numpy as np
import sklearn.neural_network as neural
import sklearn.model_selection as ms
import sklearn.metrics as metrics
import tensorflow as tf
import tensorflow.keras as keras

np.random.seed(0)
tf.random.set_seed(0)

with np.load("data/mnist/mnist.npz", allow_pickle=True) as f:
    x_train, y_train = f["x_train"], f["y_train"]
    x_test, y_test = f["x_test"], f["y_test"]
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

x_train = x_train.reshape(-1, 28*28)
x_test = x_test.reshape(-1, 28*28)

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

model = keras.Sequential()
model.add(keras.layers.Dense(600, activation="relu", input_shape=(x_train.shape[1],)))
model.add(keras.layers.Dense(400, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))

optimizer = keras.optimizers.SGD(nesterov=True, lr=1e-5)

model.compile(loss="categorical_crossentropy", metrics="accuracy", optimizer=optimizer)

model.fit(x_train, y_train, validation_split=0.2, epochs=1, batch_size=10)
print(model.evaluate(x_test, y_test))
predicted = model.predict(x_test)
model.save("/data/mnist/model.h5")

print(y_test[0], predicted[0])
print(np.argmax(predicted[0]))
