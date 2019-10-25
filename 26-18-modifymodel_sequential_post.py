import keras
import keras.applications

# Enleve les 2 derniers niveaux et en ajoute 3
model = keras.applications.vgg16.VGG16(include_top=True, weights="imagenet")

print(len(model.layers))
print(model.summary())

newModel = keras.models.Sequential()
for l in model.layers[:-1]:
    newModel.add(l)
newModel.add(keras.layers.Dense(1024,activation='relu'))
newModel.add(keras.layers.Dense(10,activation='softmax'))
print(len(newModel.layers))
newModel.build()
print(newModel.summary())

# Entraine seulement les 3 derniers layers
for l in newModel.layers[:-3]:
    l.trainable=False