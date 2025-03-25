import pandas as pd
import sklearn.preprocessing as pp
import matplotlib.pyplot as plt

dataframe = pd.read_csv("data/house/house.csv")
dataframe = dataframe[dataframe.surface < 200]

scaler = pp.MinMaxScaler()
scaler.fit(dataframe)
scaled = scaler.transform(dataframe)

plt.scatter(dataframe.surface, scaled[:,0])
plt.show()

scaler = pp.StandardScaler()
scaler.fit(dataframe)
scaled = scaler.transform(dataframe)

plt.scatter(dataframe.surface, scaled[:,0])
plt.show()

scaler = pp.RobustScaler()
scaler.fit(dataframe)
scaled = scaler.transform(dataframe)

plt.scatter(dataframe.surface, scaled[:,0])
plt.show()

