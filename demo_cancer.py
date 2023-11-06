import pandas as pd
import sklearn.linear_model as lm
import sklearn.neighbors as nn

dataframe = pd.read_csv("data/breast-cancer/data.csv", index_col="id")
y = dataframe.diagnosis
x = dataframe.drop("diagnosis", axis=1)
# model = lm.LinearRegression()
model = nn.KNeighborsClassifier(n_neighbors=3)
model.fit(x, y)
print(model.score(x, y))
