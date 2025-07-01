# Ecrire la fonction factorielle(n: int) -> int
# Idem pour is_prime(n: int) -> bool
# Tester DANS le module (main)
# Tester dans program.py

def factorielle(n: int) -> int:
    result = 1
    for i in range(1, n+1):
        result *= i
    return result


def is_prime(n: int) -> bool:
    found = False
    if n >= 2:
        for div in range(2, n):
            if n % div == 0:
                found = True
                break
    return not found


if __name__ == '__main__':
    assert 120 == factorielle(5)
    assert is_prime(7)
