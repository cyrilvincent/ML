import pandas as pd
import sklearn.linear_model as lm

df = pd.read_csv("data/heartdisease/data_cleaned_up.csv")
print(df)
target = df.num
training_set = df.drop("num", 1)
ok_set = training_set[df.num == 0]
ko_set = training_set[df.num == 1]
print(ok_set.describe().T)
print(ko_set.describe().T)
print(df.corr())

model = lm.LinearRegression()
model.fit(training_set, df.num)
predicted = model.predict(training_set)
print(model.score(training_set, df.num))

print(model.coef_, model.intercept_)

