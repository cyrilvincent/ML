import pickle
import numpy as np
import matplotlib.pyplot as plt

with open("data/mnist/rf-0.97.pickle", "rb") as f:
    model = pickle.load(f)

v784 = np.random.randint(0,256,784)
m2828 = np.reshape(v784,(28,28))

plt.imshow(m2828, cmap='gray')
plt.show()

y = model.predict(v784.reshape(1, -1))
print(y)