import pandas as pd
import matplotlib.pyplot as plt

print(pd.__version__)

dataframe = pd.read_csv("data/house/house.csv")

print(dataframe.describe())

loyers = dataframe["loyer"]
loyers = dataframe.loyer
print(loyers)

matrix_numpy = dataframe.values
print(matrix_numpy)
print(matrix_numpy.shape)

# dataframe.to_excel("data/house/house3.xlsx")

plt.scatter(dataframe.surface, dataframe.loyer)
plt.show()
print("END")
