import numpy as np
l = [1,5,9,7,8,-2,10,99]
a1 = np.array(l)
a2 = np.arange(8)
print(a2)
print(a1 + a2)
print(np.sin(a1))

mat1 = np.array([[1,2],[3,4]])
print(mat1**2)

print(mat1.shape)

v = np.array([2,4,6,8])
print(v.shape)
mat2 = v.reshape(2,2)
print(mat2)

print(mat1 + mat2)
print(mat1 * mat2)
print(mat1.dot(mat2))

v = np.arange(1000)
print(np.mean(v))
import math
print(math.sqrt(np.var(v)))

import csv
with open("house/house.csv") as f:
    reader = csv.DictReader(f)
    surfaces=[]
    loyers=[]
    for row in reader:
        surfaces.append(int(row["surface"]))
        loyers.append((int(row["loyer"])))

print(np.mean(loyers))
print(np.std(loyers))

import scipy.stats as stats
slope, intercept, r_value, p_value, std_err = stats.linregress(surfaces, loyers)
print(f"f(x) = {slope}x + {intercept}")

lx=range(10,300)
predicts = [slope * x + intercept for x in lx]

