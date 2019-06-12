import tensorflow as tf
from tensorflow import keras
import pandas as pd

data = pd.read_csv('heartdisease/data_cleaned_up.csv', na_values=['.'])
data = data.sample(frac=1)
print(data.describe)

# separate the output data (column 'num') from rest of the data
values_series = data['num']
x_data = data.pop('num')

# split input(x) and output (y) data
# for training and testing
train_x_data = data[0:100]
train_y_data = x_data[0:100]
train_x_data = train_x_data.values
train_y_data = train_y_data.values

test_x_data = data[100:]
test_y_data = x_data[100:]
test_x_data = test_x_data.values
test_y_data = test_y_data.values


print(train_x_data[0])
print(train_y_data[0])
print(train_x_data.shape)

# create model
model = keras.Sequential()

# add layers
model.add(keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(train_x_data.shape[1],)))
model.add(keras.layers.Dense(64, activation=tf.nn.relu))
model.add(keras.layers.Dense(32, activation=tf.nn.relu))
model.add(keras.layers.Dense(16, activation=tf.nn.relu))

# last layer has only two possible outcomes
# either 0 or 1 indicating not diagnosed and diagnosed respectively
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

# get summary of the model
model.summary()

# compile the model
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_x_data,
                    train_y_data,
                    epochs=40,
                    batch_size=256,
                    validation_data=(test_x_data, test_y_data),
                    verbose=1)

# evaluate the model
results = model.evaluate(test_x_data, test_y_data)
print(results)

# Create graph for acuracy and loss
history_dict = history.history
history_dict.keys()

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()   # clear figure
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
