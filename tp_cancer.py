import pandas as pd
import matplotlib.pyplot as plt

# Charger data/breast-cancer/data.csv dans un dataset
# Faire un describe sur la colonne perimeter_worst
# Sauvegarder le dataset en HTML

dataframe = pd.read_csv("data/breast-cancer/data.csv")
print(dataframe["perimeter_worst"].describe())
dataframe.to_html("data/breast-cancer/data.html")
plt.matshow(dataframe.corr())
plt.show()

