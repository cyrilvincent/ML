import numpy as np

v = np.array([0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

res = np.fft.fft(v)

print(res)

import matplotlib.pyplot as plt

plt.plot(np.real(res))
plt.show()