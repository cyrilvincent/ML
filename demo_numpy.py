import numpy as np

v1 = np.array([1,2,3,4])
print(v1.shape)
mat1 = v1.reshape(2,2)
print(mat1)

v2 = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
mat2 = v2.reshape(-1,4)
print(mat2)