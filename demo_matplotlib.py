import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-2 * np.pi, 2 * np.pi, 0.1)
y = np.sin(x)
y2 = np.cos(x)

plt.subplot(2,1,1)
plt.plot(x, y, label="sin")
plt.subplot(2,1,2)
plt.scatter(x, y2, color="red", label="cos")
plt.legend()
# plt.plot(x, y, "bo-")
# plt.plot(x, y, color="blue", marker="o", markerfacecolor="green")
# plt.scatter(x, y)
# plt.bar(x, y)
plt.savefig("demo.png")
plt.show()

