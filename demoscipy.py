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

