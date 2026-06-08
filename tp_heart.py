import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import sklearn.preprocessing as pp
import sklearn.pipeline as pipe
import sklearn.model_selection as ms
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 500)

df = pd.read_csv("data/heartdisease/data_cleaned_up.csv")
print(df.describe())

y = df["num"]
x = df.drop(["num"], axis=1)

np.random.seed(42)

xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, train_size=0.8, test_size=0.2)

model = pipe.make_pipeline(pp.PolynomialFeatures(2), lm.Ridge())

model.fit(xtrain, ytrain)

print(f"Train score: {model.score(xtrain, ytrain)}")
print(f"Test score: {model.score(xtest, ytest)}")

values = np.array([[28,1,2,130,132,0,2,185,0,0]])
predict = model.predict(values)
print(predict)