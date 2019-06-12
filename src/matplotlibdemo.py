import matplotlib.pyplot as plt

import csv
with open("house/house.csv") as f:
    reader = list(csv.DictReader(f))

loyers = [int(row["loyer"]) for row in reader]
surfaces = [int(row["surface"]) for row in reader]

import numpy as np
print(f"Loyer Moyen: {np.mean(loyers)}")
print(f"Surface Moyenne: {np.mean(surfaces)}")
print(f"Ecart type: {np.std(loyers)}")

import scipy.stats as stats
print(stats.linregress(surfaces, loyers))

loyersPredit = [41 * s - 283 for s in range(400)]

plt.scatter(surfaces,loyers)
plt.plot(range(400),loyersPredit)
plt.show()