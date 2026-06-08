import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import sklearn.preprocessing as pp
import sklearn.pipeline as pipe
import sklearn.model_selection as ms
import numpy as np

data = np.load("data/mnist/mnist.npz")
xtrain = data["x_train"]
xtest = data["x_test"]
ytrain = data["y_train"]
ytest = data["y_test"]

print(xtrain.shape)

xtrain = xtrain.reshape(-1, 28*28)
xtest = xtest.reshape(-1, 28*28)

model = None # TODO
# Apprentissage + score + predict

ypred = None #TODO

xtest = xtest.reshape(-1, 28, 28)
select = np.random.randint(xtest.shape[0], size=12)

for index, value in enumerate(select):
    plt.subplot(3, 4, index + 1)
    plt.axis("off")
    plt.imshow(xtest[value], cmap=plt.cm.gray_r)
    # plt.title(f"Predicted {ypred[value]}")
plt.show()
