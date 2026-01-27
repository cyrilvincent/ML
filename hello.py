import numpy as np
import pandas as pd

print("Hello")
print(np.__version__)

a1 = np.array([1,2,3,4])
a2 = np.array([5,6,7,8])
print(a1 + a2)
print(np.mean(a1), np.std(a1))
print(np.sin(a1))

dataframe = pd.read_csv("data/house/house.csv")
dataframe["loyer_m2"] = dataframe["loyer"] / dataframe["surface"]
dataframe = dataframe[dataframe["surface"] < 100]
dataframe.to_html("house.html")
dataframe.to_excel("house.xlsx")

# big = np.arange(1000)
# filter = big % 2 == 0
# print(filter)
# filter2 = (big % 3 == 0) & (np.sin(big) > 0)
# print(big[filter2])

