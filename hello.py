import pandas as pd

print("Hello")
print(pd.__version__)

dataframe = pd.read_csv("data/house/house.csv")
print(dataframe.describe())