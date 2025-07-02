import pandas as pd
import numpy as np

print(pd.__version__)

dataframe = pd.read_csv("data/house/house.csv")
print(dataframe)

# Afficher en x les surfaces et en y les loyers dans un scatter
# Que peut on en déduire
# Filtrer les surfaces < 180
# Reafficher
# Save
# dataframe.hist(bins=5)