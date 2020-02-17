import matplotlib.pyplot as plt
import math

x = range(1000)
y1 = [math.sin(i / 100) for i in x]
y2 = [(i / 100) * math.sin(i / 100) for i in x]
plt.subplot(121)
plt.plot(x, y1)
plt.subplot(122)
plt.plot(x, y2)
plt.show()