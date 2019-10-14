import pandas as pd

df = pd.read_excel("house/house.xlsx")
df = df[df.surface < 200]
print(df)
print(df["loyer"]/df.surface)
