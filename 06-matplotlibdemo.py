import matplotlib.pyplot as plt

import csv
with open("house/house.csv") as f:
    reader = csv.DictReader(f)
    surfaces=[]
    loyers=[]
    for row in reader:
        surfaces.append(int(row["surface"]))
        loyers.append((int(row["loyer"])))

plt.scatter(surfaces, loyers)
plt.show()