import sklearn
import pandas as pd
# pip install openpyxl
import numpy as np
import matplotlib.pyplot as plt

print("Hello")
print(sklearn.__version__)

df = pd.read_csv("data/house/house.csv")
df["loyer_m2"] = df["loyer"] / df["surface"]
print(df)
df.to_json("data/house/house.json", indent=2, index=False)
df.to_excel("data/house/house.xlsx", index=False)

print(df.describe())
print(df.corr())

v12 = np.arange(12, dtype=np.float64)
print(v12)
m34 = v12.reshape(3, 4)
print(m34)
print(m34 * 2)
print(np.cos(m34))

plt.scatter(df["surface"], df["loyer"])
plt.show()

df.hist()
plt.show()