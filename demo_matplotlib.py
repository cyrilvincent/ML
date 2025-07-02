import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-2 * np.pi, 2 * np.pi, 0.1)
y = np.sin(x)
y2 = np.cos(x)

plt.title("Sinus")
plt.scatter(x, y, color="red", label="sinus")
plt.plot(x, y2, color="blue", label="cosinus")
plt.bar(x, y2, color="green", label="cosinus")
plt.legend()
plt.show()

plt.subplot(221)
plt.scatter(x, y, color="red", label="sinus")
plt.subplot(222)
plt.plot(x, y2, color="blue", label="cosinus")
plt.subplot(223)
plt.bar(x, y2, color="blue", label="cosinus")
plt.subplot(224)
plt.plot(x, x, color="yellow", label="identity")
plt.show()



