# On importe les librairies dont on aura besoin pour ce tp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# On charge le dataset
house_data = pd.read_csv('house/house.csv')

surface_max = 300
house_data = house_data[house_data.surface < 300]
standard_dev = 2199
house_data = house_data[abs(41 * house_data.surface - 283 - house_data.loyer) < 3 * standard_dev]
print(house_data)

# On affiche le nuage de points dont on dispose
plt.plot(house_data['surface'], house_data['loyer'], 'ro', markersize=4)
plt.show()

