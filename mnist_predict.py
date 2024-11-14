import pickle
import numpy as np

with open("data/mnist/rf-97.pickle", "rb") as f:
    model = pickle.load(f)


m784 = np.random.randint(0,255,784)
m784 = m784.reshape(-1 ,784)
predicted = model.predict(m784)
print(predicted)

