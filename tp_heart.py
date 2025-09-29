import sklearn
import pandas as pd
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
import numpy as np

print(sklearn.__version__)

#1 Load data
dataframe = pd.read_csv("data/heartdisease/data_cleaned_up.csv")

#2 Make dataset
y = dataframe["num"]
x = dataframe.drop("num", axis=1)

#5 Creating the model
model = lm.LinearRegression()
# f(x) = ax + b => 2 poids

#6 Fit
model.fit(x, y)

#7 Scoring (facultatif)
score = model.score(x, y)
print(f"Score: {score:.2f}")
