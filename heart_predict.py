import sklearn
import numpy as np
import pickle

x = np.array([[28,1,2,130,132,0,2,185,0,0]])

with open("data/heartdisease/rf-80.pickle", "rb") as f:
    scaler, model = pickle.load(f)

    x = scaler.transform(x)
    ypredicted = model.predict(x)
    print(ypredicted)
