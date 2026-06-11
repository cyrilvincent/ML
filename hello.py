import pandas as pd
import matplotlib.pyplot as plt

print("Hello")
print(pd.__version__)

df = pd.read_csv("data/house/house.csv")
print(df.describe())

plt.scatter(df["surface"], df["loyer"])
plt.show()