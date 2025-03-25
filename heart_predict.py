import pickle

with open("data/heartdisease/rf-0.77.pickle", "rb") as f:
    model = pickle.load(f)

x = [[28,1,2,130,132,0,2,185,0,0,0]]
y = model.predict(x)
print(y)