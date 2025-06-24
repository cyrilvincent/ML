import pickle
import numpy as np

with open(f"data/heartdisease/rf-100.pickle", "rb") as f:
    (model, scaler) = pickle.load(f)

x = np.array([[28,1,2,130,132,0,2,185,0,0,0]])
ypredicted = model.predict(x)
print(ypredicted[0])