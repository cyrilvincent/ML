# Installer scipy
# Calculer la régression linéaire surfaces / loyers
# surfaces = surfaces[surfaces < 200] recalcul
# Charger covid-france.txt er créer les tableaux nbcas, dcs
# Afficher dans un graphique en barre (bar)
# mean, max, min, std
# Calculer la régression linéaire et voir qu'elle ne marche pas
# Facultatif : calculer la régression linéaire sur ix < 54

import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
import csv

with open("data/house/house.csv") as f:
    reader = csv.DictReader(f)
    loyers = []
    surfaces = []
    for row in reader:
        #print(float(row["loyer"]), float(row["surface"]), float(row["loyer"])/float(row["surface"]))
        loyers.append(float(row["loyer"]))
        surfaces.append(float(row["surface"]))
    loyers = np.array(loyers)
    surfaces = np.array(surfaces)

#result = stats.linregress(surfaces, loyers)
slope, intercept, r_value, p_value, std_err = stats.linregress(surfaces, loyers)
print(slope, intercept, r_value, p_value, std_err)

f = lambda x : slope * x + intercept #f(x) = 41x - 283
plt.scatter(surfaces, loyers)
plt.plot(surfaces, f(surfaces))
plt.show()

surfaces2 = surfaces[surfaces < 200]
loyers2 = loyers[surfaces < 200]
slope, intercept, r_value, p_value, std_err = stats.linregress(surfaces2, loyers2)
print(slope, intercept, r_value, p_value, std_err)
f = lambda x : slope * x + intercept #f(x) = 41x - 283
plt.scatter(surfaces2, loyers2)
plt.plot(surfaces2, f(surfaces2))
plt.show()

