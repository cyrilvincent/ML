i = 2
print("Hello World")

def isMul3(x):
    return x % 3 == 0

def isPrime(x):
    if x < 2:
        return False
    else:
        for i in range(2,x):
            if x % i == 0:
                return False
    return True

def filterEven(l):
    res = []
    for i in l:
        if i % 2 == 0:
            res.append(i)
    return res

def filterPrime(l):
    res = []
    for i in l:
        if isPrime(i):
            res.append(i)
    return res

def filterByFn(fn ,l):
    res = []
    for i in l:
        if fn(i):
            res.append(i)
    return res



f = isPrime

print(f(8))
print(isPrime(49999))
print(isPrime(8))
l = [1,2,8,9,7,99,5,88,60]
print(filterEven(l))
print(filterPrime(l))
print(filterPrime(range(1000)))
print(filterByFn(isMul3, l))
print(list(filter(isPrime, range(1000))))

def inc(x):
    return x + 1

inc = lambda x : x + 1

print(list(map(inc, l)))

