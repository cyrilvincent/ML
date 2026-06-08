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

# TP Faites une regression poly 3
