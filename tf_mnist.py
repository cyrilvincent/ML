import sklearn
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import sklearn.pipeline as pipe
import sklearn.preprocessing as pp
import sklearn.model_selection as ms
import sklearn.neighbors as n
import numpy as np
import sklearn.ensemble as rf
import keras

data = np.load("data/mnist/mnist.npz")
print(data)
xtrain = data["x_train"]
xtest = data["x_test"]
ytrain = data["y_train"]
ytest = data["y_test"]

print(xtest.shape)

np.random.seed(42)

# xtrain = xtrain.reshape(-1, 28*28).astype(np.float64)
# xtest = xtest.reshape(-1, 28*28).astype(np.float64)
xtrain = xtrain.astype(np.float64)
xtest = xtest.astype(np.float64)
# print(xtest.shape)

xtrain /= 255
xtest /= 255

ytrain = keras.utils.to_categorical(ytrain)
ytest = keras.utils.to_categorical(ytest)
model = keras.Sequential()
model.add(keras.layers.Input((28,28,1)))
model.add(keras.layers.ZeroPadding2D((1,1)))
model.add(keras.layers.Convolution2D(8, (3, 3), activation='relu'))
model.add(keras.layers.ReLU())
model.add(keras.layers.MaxPooling2D((2,2), strides=(2,2)))

model.add(keras.layers.ZeroPadding2D((1,1)))
model.add(keras.layers.Convolution2D(16, (3, 3), activation='relu'))
model.add(keras.layers.ReLU())
model.add(keras.layers.MaxPooling2D((2,2), strides=(2,2)))

model.add(keras.layers.ZeroPadding2D((1,1)))
model.add(keras.layers.Convolution2D(16, (3, 3), activation='relu'))
model.add(keras.layers.ReLU())
model.add(keras.layers.MaxPooling2D((2,2), strides=(2,2)))
model.add(keras.layers.Flatten())
# model.add(keras.layers.Dense(600, activation="relu"))
# model.add(keras.layers.Dropout(0.5))
# model.add(keras.layers.Dense(400, activation="relu"))
# model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(200, activation="relu"))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(10, activation="softmax"))

model.summary()

model.compile(optimizer="rmsprop", metrics=["accuracy"], loss="categorical_crossentropy")
model.fit(xtrain, ytrain, epochs=10, batch_size=10, validation_split=0.2)
print(model.evaluate(xtest, ytest))
model.save("data/mnist/mlp.keras")

ypred = np.argmax(model.predict(xtest))


xtest = xtest.reshape(-1, 28, 28)
select = np.random.randint(xtest.shape[0], size=12)

for index, value in enumerate(select):
    plt.subplot(3,4, index + 1)
    plt.axis("off")
    plt.imshow(xtest[value], cmap=plt.cm.gray_r)
    plt.title(f"Predicted {ypred[value]}")
plt.show()

errors = ytest != ypred
xerrors = xtest[errors]
yerrors = ypred[errors]

select = np.random.randint(xerrors.shape[0], size=12)

for index, value in enumerate(select):
    plt.subplot(3,4, index + 1)
    plt.axis("off")
    plt.imshow(xerrors[value], cmap=plt.cm.gray_r)
    plt.title(f"Predicted {yerrors[value]}")
plt.show()

plt.imshow(model.feature_importances_.reshape(28, 28))
plt.show()



