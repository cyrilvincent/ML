# 1 Refaire fonctionner house_regression.py

import sklearn.linear_model as lm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
dataframe = pd.read_csv("data/heartdisease/data_cleaned_up.csv")

print(dataframe.describe())

y = dataframe["num"]
x = dataframe.drop(["num"], axis=1)

print(dataframe.corr())