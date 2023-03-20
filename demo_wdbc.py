import pandas as pd

df = pd.read_csv("data/breast-cancer/data.csv")
target = df.diagnosis
training_set = df.drop(["id", "diagnosis"], 1)
ok_set = training_set[df.diagnosis == 0]
ko_set = training_set[df.diagnosis == 1]
print(ok_set.describe().T)
print(ko_set.describe().T)