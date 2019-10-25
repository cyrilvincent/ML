import keras.applications
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt

model = keras.applications.vgg16.VGG16(include_top=False, weights="imagenet")
print(len(model.layers))
print(model.summary())

filters, biases = model.layers[1].get_weights()
# normalize filter values to 0-1 so we can visualize them
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)
# plot first few filters
n_filters, ix = 6, 1
for i in range(n_filters):
	# get the filter
	f = filters[:, :, :, i]
	# plot each channel separately
	for j in range(3):
		# specify subplot and turn of axis
		ax = plt.subplot(n_filters, 3, ix)
		ax.set_xticks([])
		ax.set_yticks([])
		# plot filter channel in grayscale
		plt.imshow(f[:, :, j], cmap='gray')
		ix += 1
# show the figure
plt.show()
# 1er conv layer
# 6 permiers filtres
# 1 ligne = 1 filtre
# 1 colonne = un canal (RVB)
"""We can see that in some cases, the filter is the same across the channels (the first row), and in others, the filters differ (the last row).
The dark squares indicate small or inhibitory weights and the light squares represent large or excitatory weights. Using this intuition, we can see that the filters on the first row detect a gradient from light in the top left to dark in the bottom right"""

image = load_img('img/mug.jpg', target_size=(224, 224))
image = img_to_array(image)
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
image = keras.applications.vgg16.preprocess_input(image)
cnnoutput = model.predict(image)
print(cnnoutput.shape) #(1,7,7,512)

ixs = [2, 5, 9, 13, 17] # Show only conv layers after zero padding
outputs = [model.layers[i].output for i in ixs]
model = keras.Model(inputs=model.inputs, outputs=outputs)
cnnoutput = model.predict(image)
square = 8

for fmap in cnnoutput:
    # plot all 64 maps in an 8x8 squares
    ix = 1
    for _ in range(square):
        for _ in range(square):
            ax = plt.subplot(square, square, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.imshow(fmap[0, :, :, ix-1], cmap='gray')
            ix += 1
    plt.show()