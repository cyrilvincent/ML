print("Hello World!")

a=1


def add(i:int,j:int) -> int:
    return i+j

def boucle(nb):
    for i in range(nb):
        print(i)

def sum(l):
    total = 0
    for i in l:
        total += i  # total = total + i
    return total

def filter_even(l):
    result = []
    for i in l:
        if i % 2 == 0:
            result.append(i)
    return result



toto = [1,2,6,99,98,5,4]
print(len(toto))
for i in toto:
    print(i)
print(sum(toto))
print(filter_even(toto))

# result = add(3.14,2)
# print(result)
# boucle(100)
n = 5
for i in range(1, n+1):
    print(i)