import tensorflow.keras as keras

model = keras.models.load_model('data/dogsvscats/vgg16model-small.h5')
newModel = keras.models.Sequential()

for layer in model.layers[:-1]:
    newModel.add(layer)
    layer.trainable = False

newModel.add(keras.layers.Dense(3, name="dense3"))
newModel.add(keras.layers.Activation('softmax'))

newModel.summary()

newModel.compile(loss='categorical_crossentropy',
              optimizer="rmsprop",
              metrics=['accuracy'])

trainset = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255, validation_split=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

batchSize = 16

trainGenerator = trainset.flow_from_directory(
        'data/dogsvscats/small/train',
        target_size=(224, 224),
        subset='training',
        class_mode="categorical",
        batch_size=batchSize)

validationGenerator = trainset.flow_from_directory(
        'data/dogsvscats/small/train',
        target_size=(224, 224),
        class_mode="categorical",
        subset = 'validation',
        batch_size=batchSize)


newModel.fit(
        trainGenerator,
        epochs=30,
        validation_data=validationGenerator,
)

newModel.save('data/dogsvscats/vgg16model-cows.h5')


