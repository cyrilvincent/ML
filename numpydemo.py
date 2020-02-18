import numpy as np
v1 = np.array([1,2,3,4])
v2 = np.arange(4)
print(v1*v2)
print(v1.shape)

mat1 = np.array([[1,2],[3,4]])
print(mat1)
print(mat1.shape)

print(v1.reshape(2,2))
print(mat1.reshape(4))
print(np.linalg.inv(mat1))

print(np.array([1,2]) * np.array([3,4]))
print(np.array([1,2]).dot(np.array([3,4])))