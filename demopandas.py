import pandas as pd
import matplotlib.pyplot as plt
import sqlite3

# with sqlite3.connect("data/house/house.db3") as conn:
#     dataset = pd.read_sql('select loyer,surface from house', conn)

#pd.read_excel("data/house/house.xslx")*


dataset = pd.read_csv("data/house/house.csv")
plt.scatter(dataset.surface, dataset.loyer)
plt.show()

import numpy as np
x = np.nan
