import pickle
import numpy as np
import matplotlib.pyplot as plt

with open("data/mnist/rf-0.971.pkl", "rb") as f:
    result = pickle.load(f)
    model = result

data = np.load("data/mnist/mnist.npz")
xtest = data["x_test"][:12]

ypred = model.predict(xtest.reshape(-1, 28*28))

for index, value in enumerate(xtest):
    plt.subplot(3, 4, index + 1)
    plt.axis("off")
    plt.imshow(value, cmap=plt.cm.gray_r)
    plt.title(f"Predicted {ypred[index]}")
plt.show()
