import numpy as np
print(np.__version__)

v1 = np.array([1,2,3,4])
print(v1.dtype)
v2 = np.array([1.,2,3,4])
print(v2.dtype)
v3 = np.array([1,2,3,4], dtype=np.float64)
print(v3.dtype)
print(v1.astype(np.float64).dtype)
v4 = np.array([3.14, 9.99])
print(v4.astype(np.int64))
v5 = np.array([1,2,3,4,255], dtype=np.uint8) # Octet
print(v5)
print("V5+1", v5 + 1)

v6 = np.arange(1,10,2)
print(v6)
v7 = np.linspace(0,10,101)
print(v7)

print(np.random.seed(42))
rnd = np.random.rand(10)
print(rnd)
rnd2 = np.random.randint(0,10000000,5)
print(rnd2)

v1 = np.array([1,2,3,4])
v2 = np.array([5,6,7,8])
print(v1 * 2 + 1)
v3 = v1 + v2
print(v3 ** 2)
print(np.cos(v3))
print(np.log10(np.array([10,100,1000])))

print(np.sum(v1))

# v4 = np.array([1,2,3])
# print(v1 + v4)

print(v1.ndim, v1.size, v1.shape)


print(rnd)
