import pandas as pd
import numpy as np
import sklearn
import sklearn.linear_model as lm
import matplotlib.pyplot as plt

df = pd.read_csv("data/heartdisease/data_cleaned_up.csv")
y = df.num
x = df.drop("num", axis=1)
print(y.describe())
print(x.describe().T)

ok = df[df.num == 0]
ko = df[df.num == 1]
xok = ok.drop("num", axis=1)
xko = ko.drop("num", axis=1)

print(xok.describe().T)
print(xko.describe().T)

# Instancier le LinearModel
# Fit sans values.reshape
# Predict
# afficher coef et intercept
# Score