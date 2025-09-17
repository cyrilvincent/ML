import numpy as np

# Copier depuis le zip de ce matin le répertoire data/* à la racine de votre projet
data = np.load("data/house/house_filtre.npz")
loyers = data["loyers_filtre"]
surfaces = data["surfaces_filtre"]

# Créer le tableau surface_m2
# Créer le calcul (cos(loyers *2 +55))²
# Créer le tableau loyer_log10
# Afficher le loyer min et max


