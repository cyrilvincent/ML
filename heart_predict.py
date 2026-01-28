import pickle

with open("data/heartdisease/rf-0.83.pickle", "rb") as f:
    model, scaler = pickle.load(f)

data = [[28,1,2,130,132,0,2,185,0,0]]
data = scaler.transform(data)
predicted = model.predict(data)
print(predicted[0])
