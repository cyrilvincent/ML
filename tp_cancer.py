import pandas as pd
import sklearn.neighbors as neighbors

dataframe = pd.read_csv("data/breast-cancer/data.csv", index_col="id")
print(dataframe)
y = dataframe.diagnosis
x = dataframe.drop("diagnosis", axis=1)

