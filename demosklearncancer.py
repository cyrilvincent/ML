import pandas as pd
import sklearn.linear_model as linearmodel

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer() # more info : https://goo.gl/U2Uwz2

#input
x=cancer['data']
y=cancer['target']
model = linearmodel.LinearRegression()

model.fit(x,y)
print(model.score(x,y))
print(model.coef_, model.intercept_)
print(model.predict(x) < 0.5)

