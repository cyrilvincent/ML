import csv

with open("data/house/house.csv") as f:
    reader = list(csv.DictReader(f))
    # for row in reader:
    #     print(row["loyer"], row["surface"], float(row["loyer"]) / float(row["surface"]))

    x = [float(row["surface"]) for row in reader]
    y = [float(row["loyer"]) for row in reader]
    print(x)
    print(y)

    # 1/ Afficher le nuage de point x / y
    # 2/ Critiquer ce nuage de point
    # 3/ Trouver un modèle mathématique
    # 4/ Calculer le loyer / m² moyen