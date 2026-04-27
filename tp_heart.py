import numpy as np
import pandas as pd
import sklearn.linear_model as lm
import sklearn.preprocessing as pp
import sklearn.pipeline as pipe
import sklearn.model_selection as ms

df = pd.read_csv("data/heartdisease/data_cleaned_up.csv")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

y = df["num"]
x = df.drop(["num"], axis=1)

print(df.corr())

# train_test_split
# fit
# predict
# score