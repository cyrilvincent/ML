import numpy as np

a1 = np.array([1,2,3,5,8,10,12,18,50,99,100])
a2 = np.arange(10)

print(a1)
print(a2)
print(np.sin(a1))

m1 = np.array([[1,2],[3,4]])
print(m1)
print(m1.shape)
v1 = np.array([21.1,22,22.2,23])
m2 = v1.reshape(2,2)
print(m2)
print(np.dot(m1, m2))
print(np.sum(m1,0))