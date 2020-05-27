import matplotlib.pyplot as plt
import math
import csv

# l = range(1000)
# for i in l:
#     print(i)

# x = range(10)
# y = range(10)
#
# #f = lambda x : math.sin(x) # f(x) = sin(x)
#
# plt.bar(x,y)
# plt.show()

with open("data/house/house.csv") as f:
    reader = csv.DictReader(f)
    loyers = []
    surfaces = []
    for row in reader:
        #print(float(row["loyer"]), float(row["surface"]), float(row["loyer"])/float(row["surface"]))
        loyers.append(float(row["loyer"]))
        surfaces.append(float(row["surface"]))
print(loyers)
print(surfaces)
print(max(loyers))
print(min(loyers))
print(len(loyers))
print(len(surfaces))

plt.scatter(surfaces, loyers)
plt.show()






