# Installer numpy
# Créer le tableau loyers et surfaces
# Créer le tableau loyerperm2 = loyers / surfaces
# Calculer min ,max, moyenne, ecart-type, median, normalisations = (x - moyenne) / std

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
loyerperm2 = loyers / surfaces
mean = np.mean(loyerperm2)
std = np.std(loyerperm2)
print(np.min(loyerperm2), np.max(loyerperm2), mean, std, np.median(loyerperm2))
normalisations = (loyerperm2 - mean) / std
print(np.min(normalisations), np.max(normalisations), np.mean(normalisations), np.std(normalisations), np.median(normalisations))
print(len(normalisations[normalisations > 0]))
print(len(normalisations[normalisations < 0]))

plt.scatter(surfaces, loyers)
plt.show()