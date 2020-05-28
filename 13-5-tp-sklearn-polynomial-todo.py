import pandas as pd
import sklearn.linear_model as sklm

data = pd.read_csv('data/house/house.csv')
model = sklm.LinearRegression()
x = data["surface"].values.reshape(-1,1)
y = data["loyer"]

