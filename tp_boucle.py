# 1 Afficher les nombres pairs < 100
# 2 Saisir un chiffre et afficher sa factorielle, ex 5! = 1*2*3*4*5 = 120
# 3 Bonus : Saisir un chiffre et afficher s'il est premier ou non, tout nombre n >= 2 est premier sauf s'il possède un diviseur entre 2 et n-1
# Rappel 8 % 2 == 0

for i in range(0, 101, 2):
    print(i)

n = int(input("n: "))
# result = 1
# for i in range(1, n+1):
#     result *= i
# print(f"{n}!={result}")

found = False
if n >= 2:
    for div in range(2, n):
        if n % div == 0:
            print("Non premier")
            found = True
            break
    if not found:
        print("Premier")
else:
    print("Non premier")


