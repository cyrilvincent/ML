from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D

def VGG_16(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    model.load_weights("vgg16_weights.h5")

def calcNbW(depths = [32], hiddens = [512], out = 10, input = 224, drop = 2, dim = 3, window = 9):
    res = 0
    prev = 1
    for d in depths:
        res += prev*dim*window*d
        prev = d
    res += (input/(drop**len(depths)))**2*prev*hiddens[0]
    prev = hiddens[0]
    for h in hiddens[1:]:
        res += h * prev
        prev = h
    res += out * prev
    return int(res)

def calcNbIter(depths = [32], hiddens = [512], out = 10, input = 224, drop = 2, dim = 3, window = 9):
    res = 0
    prev = 1
    i = 1
    for d in depths:
        res += prev*dim*window*d*(input/i)**2
        i *= drop
        prev = d
    res += (input/(drop**len(depths)))**2*prev*hiddens[0]
    prev = hiddens[0]
    for h in hiddens[1:]:
        res += h * prev
        prev = h
    res += out * prev
    return int(res)

vgg16w = calcNbW([32,64,128,256,512],[4096,4096,1024],1000)
print(f"Nb kweigths={int(vgg16w/1000)}")
vgg16i = calcNbIter([32,64,128,256,512],[4096,4096,1024],1000)
print(f"Nb Miter per CPU inference={int(vgg16i/1e6)}")
print(f"Nb Miter per GPU inference for batch={28*28} and a good GPU={int((vgg16i / (28 * 28))/1e6)}") #Difficile à calculer dépend de la taille de l(image du batch et des GPUs