import numpy as np
import matplotlib.pyplot as plt

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

print(np.mean(loyers), np.std(loyers), np.median(loyers), np.quantile(loyers, [0.25, 0.75]))
print(np.mean(surfaces), np.std(surfaces), np.median(surfaces), np.quantile(surfaces, [0.25, 0.75]))
print(loyers[loyers < 1000])
print(loyers[surfaces > 100])
print(loyers[(surfaces > 100) & (loyers < 2500) ])
# Stats sur les loyers et surfaces
# Filtrer les loyers < 1000
# Afficher les loyers dont les surfaces > 100
# Afficher les loyers dont les surfaces > 100 & loyers < 3000

mat22 = np.array([[1,2],[3,4]])
mat22[0,1]
print(mat22)
# print(np.sum(mat22))
print(np.sum(mat22, axis=0)) # NP ROW FIRST
print(np.sum(mat22, axis=1)) # COLUMN

mat12 = np.array([[1,2]])
mat21 = np.array([[1],[2]])
print(mat12.shape)
print(mat21.shape)
v2 = np.array([1,2])
print(v2.shape)
print(mat12)
print(mat21)

# Afficher dans matplotlib le nuage de point des x = surfaces, y = loyers

plt.scatter(surfaces, loyers)
plt.show()

for loyer in loyers:
    print(loyer)

for i in range(len(loyers)):
    print(loyers[i])


