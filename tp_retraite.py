# Age de la retraite taux plein
# > 67 ans : ok
# < 64 ans : KO
# age = input
# nb_trimestre = input
# 64 + (172 - nb_trimestre) / 4

age = input("Age: ")
age = int(age)
if age > 67:
    print("Taux plein")
elif age < 64:
    print("Pas de retraite")
else:
    nb_trimestre = int(input("Nb trimestre: "))
    age_retraite = 64 + (172 - nb_trimestre) / 4
    if age_retraite > 67:
        age_retraite = 67
    print(f"Age de la retraite à taux plein: {age_retraite:.1f}")


