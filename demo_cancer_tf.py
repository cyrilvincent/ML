import tensorflow as tf
import pandas
import sklearn.model_selection as ms
import sklearn.preprocessing as pp


tf.random.set_seed(1)

dataframe = pandas.read_csv("data/breast-cancer/data.csv", index_col="id")
y = dataframe.diagnosis
x = dataframe.drop(["diagnosis"], axis=1)

xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, test_size=0.2, train_size=0.8)

scaler = pp.RobustScaler()
scaler.fit(xtrain)
xtrain = scaler.transform(xtrain)
xtest = scaler.transform(xtest)

ytrain = tf.keras.utils.to_categorical(ytrain)
ytest = tf.keras.utils.to_categorical(ytest)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(30, activation=tf.nn.relu,
                       input_shape=(x.shape[1],)),
    tf.keras.layers.Dense(30, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(30, activation=tf.nn.relu),
    tf.keras.layers.Dense(2, activation=tf.nn.softmax)
  ])

# 570 data => 2 à 3 layers
# 30 colonnes => == =>
# 1 résultat
# Tester en V, avec 1 layer, avec 5 layers, 10, 20

model.compile(loss="categorical_crossentropy", optimizer="rmsprop",metrics=['accuracy'])
model.summary()

hist = model.fit(xtrain, ytrain, epochs=100, batch_size=1, validation_split=0.2)

eval = model.evaluate(xtest, ytest)
predicted = model.predict(xtest)
print(eval)
model.save(f"data/breast-cancer/mlp-{int(eval[1]*100)}.h5")
print(predicted[0])

import matplotlib.pyplot as plt
f, ax = plt.subplots()
ax.plot([None] + hist.history['accuracy'], 'o-')
ax.legend(['Train accuracy'], loc = 0)
ax.set_title('Accuracy per Epoch')
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
plt.show()
