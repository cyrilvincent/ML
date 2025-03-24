import pandas as pd
import sklearn.linear_model as lm

dataframe = pd.read_csv("data/heartdisease/data_cleaned_up.csv")

y = dataframe.num
x = dataframe.drop("num", axis=1)

# LinearModel
# Fit
# Predict
# Score
