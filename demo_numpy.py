import numpy as np

print(np.__version__)

a1 = np.array([1,2,3,4])
print(a1)
print(a1 ** 2)
print(np.sin(a1) ** 2)

a2 = np.array([5,6,7,8])
print(a1 * a2)
print(np.dot(a1, a2)) # (1+2)*(3+4)=1*3+1*4+2*3+2*4

print(np.mean(a1), np.std(a1))

print(a1.shape, a1.ndim, a1.size, a1.dtype)
m1 = np.array([[1,2],[3,4]])
print(m1)
print(m1.shape, m1.ndim, m1.size, m1.dtype)

for i in a1:
    print(i)

print(a1[1], a1[-1])
a3 = np.arange(10,20)
n = 2

m2 = np.arange(100).reshape(10,10)
print(m2[2:-2,3:-3])


print(a3, a3[3:7], a3[1:-1], a3[n:-n])
print(a3[a3 < 5], a3[a3 % 2 == 0], a3[(a3 % 2 == 0) & (a3 < 5)])

a4 = np.array([55,4,99,88,77,6,14,28,55,54])

print(a3, a3[a4 < 50])
print(a4, a4[a4 < 50])
print(a4 < 50)
a4 = a4[a4 < 50]
filter = (a4 < 50)
# print(a3[filter])
# print(a4[filter])

# Créer un tableau de 100 entiers
# Filtrer les entiers multiple de 3 et < 50
# Récupérer le résultat et monter au carré et appliquer un sinus
a100 = np.arange(100)
filter = ((a100 % 3 == 0) & (a100 < 50))
result = a100[filter]
print(result)
print(np.sin(np.pow(result, 2)))
