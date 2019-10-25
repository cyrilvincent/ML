import keras
import keras.applications

# Enleve les 2 derniers niveaux et en ajoute 3
model = keras.applications.vgg16.VGG16(include_top=True, weights="imagenet")

print(len(model.layers))
print(model.summary())

newModel = keras.models.Sequential()
for l in model.layers[:-3]:
    newModel.add(l)
newModel.build()
print(newModel.summary())

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
image = load_img('img/mug.jpg', target_size=(224, 224))
image = img_to_array(image)
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
image = keras.applications.vgg16.preprocess_input(image)
flattenoutput = newModel.predict(image)
flattenoutput = flattenoutput.reshape(25088)
print(list(flattenoutput))

# Flatten = Input of MLP c'st pour celà qu'il est dans le top
# Nb Connexion = 102764544 = 4096 * (25088 + 1)

# peut être mis en entrée de KNN RF XGBoost ...