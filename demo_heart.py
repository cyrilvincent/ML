import pandas as pd

df = pd.read_csv("data/heartdisease/data_cleaned_up.csv")
print(df)
target = df.num
training_set = df.drop("num", 1)
ok_set = training_set[df.num == 0]
ko_set = training_set[df.num == 1]
print(ok_set.describe().T)
print(ko_set.describe().T)
print(df.corr())
