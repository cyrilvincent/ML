# Créer une liste de 10 entiers "en dur" toto = [1,2,6,99,98,5,4]
# Créer la fonction display qui prend en paramètre une liste et affiche le contenu
# Créer le fonction factorielle qui retourne la factorielle de n : factorielle(5) => 120 => 5*4*3*2*1 => range(1,6)
# Créer la fonction qui calcul la moyenne des elements d'une liste : len()
# Bonus : créer le foncton is_prime(n) qui retourne True ou False si n est premier



def display(l):
    for i in l:
        print(i)

def facto(n):
    result = 1
    for i in range(2, n+1):
        result = result * i
    return result

def is_prime(n):
    if n < 2:
        return False
    for div in range(2, n):
        if n % div == 0:
            return False
    return True

def avg(l):
    total = 0
    for i in l:
        total += i
    return total / len(l)

l = [1,2,6,99,98,5,4,8,9,10]
display(l)
print(facto(10))
print(avg(l))
print(is_prime(7))
print(is_prime(100000002))
print(is_prime(1223))