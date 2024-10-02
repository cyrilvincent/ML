import pickle
import numpy as np

with open("data/heartdisease/rf.pickle", "rb") as f:
    model = pickle.load(f)
    data = np.array([[28,1,2,130,132,0,2,185,0,0]])
    ypred = model.predict(data)
    print(ypred[0])