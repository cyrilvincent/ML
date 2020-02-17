import math as m

def add(i,j=0):
    return i + j

print(add(i = 2, j = 3))

def tanh():
    return "x"

m.tanh(0)
print(m.sin(0))
print("Hello World")

i = 1

def isPrime(x):
    if x < 2:
        return False
    else:
        for div in range(2, int(x ** 0.5 + 1)):
            if x % div == 0:
                return False
        return True

print(isPrime(6113))
print(isPrime(6114))
l = [2,4,9,8,99,-2,0]
# min, max, sum, len
print(sum(l) / len(l))
for val in l:
    print(val)

i = 3
j = i
j+=1
print(i, j)

i = [1,2,3]
j = i
j.append(4)
print(i, j)

i = [1,2,3]
j = list(i)
j.append(4)
print(i, j)