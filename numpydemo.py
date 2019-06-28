import numpy as np

v1 = np.array([1,2,3,4,5])
print(type(v1))
print(v1.shape)

mat1 = np.array([[1,2,3,10],[4,5,6,11],[7,8,9,12]])
print(mat1)
print(mat1.shape)

v2 = np.arange(9).astype(float).reshape(3,3)
print(v2)