import pandas
import pandas as pd
import matplotlib.pyplot as plt

print("Hello")
print(pandas.__version__)

df = pd.read_csv("data/house/house.csv")
print(df)
print(df.describe())
print(df.corr())

plt.scatter(df["surface"], df["loyer"])
plt.show()