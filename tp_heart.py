import pandas as pd
import sklearn.linear_model as lm
import numpy as np



dataframe = pd.read_csv("data/heartdisease/data_cleaned_up.csv")

y = dataframe.num
x = dataframe.drop("num", axis=1)

model = lm.LinearRegression()

model.fit(x, y)

score = model.score(x, y)
print(score)

predicted = model.predict(x)
print(predicted)

# LinearModel
# Fit
# Predict
# Score
