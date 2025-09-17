import numpy as np

# Copier depuis le zip de ce matin le répertoire data/* à la racine de votre projet
data = np.load("data/house/house_filtre.npz")
loyers = data["loyers_filtre"]
surfaces = data["surfaces_filtre"]

# Créer le tableau surface_m2 :
# Créer le calcul (cos(loyers *2 +55))²
# Créer le tableau loyer_log10
# Afficher le loyer min et max

surface_m2 = loyers / surfaces
print(surface_m2)
print(np.cos(loyers * 2 + 55) ** 2)
tableau_log10 = np.log10(loyers)
print(tableau_log10)
print("max:", np.max(loyers), "min:", np.min(loyers))

# Stats sur les loyers et surfaces
# Filtrer les loyers < 1000
# Afficher les loyers dont les surfaces > 100
# Afficher les loyers dont les surfaces > 100 & loyers > 1000


