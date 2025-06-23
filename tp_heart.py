import pandas as pd
import sklearn.linear_model as lm

pd.set_option('display.max_columns', None)
dataframe = pd.read_csv("data/heartdisease/data_cleaned_up.csv")
print(dataframe.describe())

y = dataframe.num
x = dataframe.drop("num", axis=1)

