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
print(loyers)
print(surfaces)
print(np.mean(loyers))
print(np.std(loyers))
print(np.var(loyers))
loyerperm2 = loyers / surfaces

plt.scatter(surfaces, loyers)
plt.show()






