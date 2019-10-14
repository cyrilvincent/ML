import matplotlib.pyplot as plt
import math
lx = range(1000)

def sinx(x):
    return math.sin(x) / 100

ly = list(map(sinx, lx))

ly = [math.sin(x / 100) * 1000 for x in lx]
plt.plot(lx, ly)
plt.show()

import csv
with open("house/house.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(int(row["loyer"]) / int(row["surface"]))