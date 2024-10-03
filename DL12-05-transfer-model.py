import tensorflow as tf
import tensorflow.keras as keras

model = keras.applications.vgg16.VGG16(include_top=False, weights="imagenet", input_shape=(224, 224, 3))

for layer in model.layers:
    layer.trainable = False

model2 = keras.applications.mobilenet_v2.MobileNetV2(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
for layer in model2.layers:
    layer.trainable = False

x = model.output
x2 = model2.output
x = keras.layers.Flatten()(x, x2)
x = keras.layers.Dense(30, activation="relu")(x)
x = keras.layers.Dense(30, activation="relu")(x)
x = keras.layers.Dense(30, activation="relu")(x)
x = keras.layers.Dense(1, activation="sigmoid")(x)

model = keras.models.Model(inputs=model.input, outputs=x)

model.summary()



