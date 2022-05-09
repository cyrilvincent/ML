import matplotlib.pyplot as plt
import matplotlib
import math

print(matplotlib.__version__)

x = range(1000)
y = range(1000)

def calc_y():
    res = []
    for i in range(1000):
        res.append(math.sin(i / 100))
    return res

y = calc_y()
plt.scatter(x, y)
plt.show()
# comment
