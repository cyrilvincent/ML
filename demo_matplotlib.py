import matplotlib.pyplot as plt
import pandas as pd

# plt.bar(range(100), range(100))
# plt.show()

df = pd.read_csv("data/house/house.csv")
print(df)

plt.scatter(df.surface, df.loyer)
plt.show()
