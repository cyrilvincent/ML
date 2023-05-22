import matplotlib
import matplotlib.pyplot as plt
import numpy as np

print("Hello world")
print(matplotlib.__version__)

x = np.arange(0, 1000)
y = np.sin(x / 200)

plt.plot(x, y)
plt.show()