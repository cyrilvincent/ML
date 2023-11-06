import pandas as pd
import sklearn.linear_model as lm

dataframe = pd.read_csv("data/breast-cancer/data.csv", index_col="id")
y = dataframe.diagnosis
x = dataframe.drop("diagnosis", axis=1)
model = lm.LinearRegression()
model.fit(x, y)
print(model.coef_, model.intercept_)
print(model.score(x, y))
print(dataframe)