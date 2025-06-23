import pandas as pd
import sklearn.linear_model as lm
import sklearn.neighbors as nn
import numpy as np

pd.set_option('display.max_columns', None)
dataframe = pd.read_csv("data/heartdisease/data_cleaned_up.csv")
print(dataframe.describe())

y = dataframe.num
x = dataframe.drop("num", axis=1)

np.random.seed(42)
# model = lm.LinearRegression()
for i in range(3,12):
    model = nn.KNeighborsClassifier(n_neighbors=i)
    model.fit(x, y)
    print(model.score(x, y))



