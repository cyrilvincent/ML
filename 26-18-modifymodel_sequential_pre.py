import keras
import keras.applications


# Ajoute un niveau
model = keras.applications.vgg16.VGG16(include_top=True, weights="imagenet")
print(len(model.layers))
print(model.summary())

newModel = keras.models.Sequential()
newModel.add(keras.layers.convolutional.Conv2D(32, (3, 3), activation='relu', input_shape=(599, 599, 3)))
for l in model.layers[1:]:
    newModel.add(l)

newModel.summary()