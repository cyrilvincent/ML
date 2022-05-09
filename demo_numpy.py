import numpy as np

tab1 = np.array([1,2,3,4,5])
tab2 = np.array([6,7,8,9,10])
print(tab1 * tab2)

mat1 = np.array([[1,2,3],[4,5,6]])
print(mat1)
print(mat1.shape)
v1 = mat1.reshape(6)
print(v1)
mat2 = mat1.reshape(3,2)
print(mat2)