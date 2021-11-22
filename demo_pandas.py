import pandas as pd
import matplotlib.pyplot as plt

print(pd.__version__)

dataframe = pd.read_csv("data/house/house.csv")
# dataframe = pd.read_excel("data/house/house.xlsx")
print(dataframe.describe())
print(dataframe)
print(dataframe["surface"])
print(dataframe.loyer)
dataframe.to_html("data/house/house.html")

plt.scatter(dataframe.surface, dataframe.loyer)
plt.show()

# Mini TP
# Lire house.csv avzec Pandas
# Afficher le describe
# Calculer loyer par m² => describe
# Afficher dans matplotlib