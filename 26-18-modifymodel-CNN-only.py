import keras.applications
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

model = keras.applications.vgg16.VGG16(include_top=False, weights="imagenet")
print(len(model.layers))
print(model.summary())

image = load_img('img/mug.jpg', target_size=(224, 224))
image = img_to_array(image)
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
image = keras.applications.vgg16.preprocess_input(image)
cnnoutput = model.predict(image)
print(cnnoutput.shape) #(1,7,7,512)
flatten = []
for i in range(7):
    for j in range(7):
        for k in range(512):
            flatten.append(cnnoutput[0][i][j][k])
print(flatten)
