for i in range(10):
    print(i)

for i in range(3, 14, 2):
    print(i)

for i in range (100, 0, -3):
    print(i)

for i in range(1, 11):
    for j in range(1, 11):
        print(f"{i}*{j}={i*j}")

for i in range(10):
    if i % 2 == 0:
        print(i)
    if i == 5:
        break
