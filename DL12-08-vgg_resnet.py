import tensorflow as tf
import tensorflow.keras as keras

model_vgg = keras.applications.vgg16.VGG16(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
model_resnet = keras.applications.resnet_v2.ResNet152V2(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
for layer in model_vgg.layers:
    layer.trainable = False

x = model_vgg.output
x = keras.layers.Flatten()(x)
y = model_resnet.output
y = keras.layers.Flatten()(y)
x = keras.layers.Dense(256, activation="relu")(x, y)
x = keras.layers.Dropout(0.2)(x)
x = keras.layers.Dense(128, activation="relu")(x)
x = keras.layers.Dropout(0.2)(x)
x = keras.layers.Dense(30, activation="relu")(x)
x = keras.layers.Dropout(0.2)(x)
x = keras.layers.Dense(1, activation="sigmoid")(x)

model_vgg = keras.models.Model(inputs=model_vgg.input, outputs=x)

model_vgg.summary()

model_vgg.compile(loss='binary_crossentropy',
                  optimizer="rmsprop",
                  metrics=['accuracy'])

trainset = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

batchSize = 16

trainGenerator = trainset.flow_from_directory(
        'data/dogsvscats/small/train',
        target_size=(224, 224),
        subset = 'training',
        class_mode="binary",
        batch_size=batchSize)

validationGenerator = trainset.flow_from_directory(
        'data/dogsvscats/small/validation',
        target_size=(224, 224),
        class_mode="binary",
        subset = 'validation',
        batch_size=batchSize)

model_vgg.fit(
        trainGenerator,
        epochs=5,
        validation_data=validationGenerator,
)

model_vgg.save('data/dogsvscats/vgg16model-small.h5')




