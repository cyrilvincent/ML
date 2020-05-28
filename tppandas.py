# Installer pandas
# Import house.csv
# Refaire l'exercice précédent et retrouver les mêmes data
# Importer wdbc en s'inspirant de 12-05-pandas-cancer
# Calculer le rayon des cellules X[:,0] calculer moyenne ecart type
# Calculer moyenne ecart type pour les cellules cancereuses
# Calculer moyenne ecart type pour les cellules saines

from sklearn.datasets import load_breast_cancer
import numpy as np
cancer = load_breast_cancer() # more info : https://goo.gl/U2Uwz2

#input
X=cancer['data']
y=cancer['target']

print(cancer.feature_names)
print(X.shape) #569 * 30
print(y.shape) #569

rayons = X[:,0]
print(np.mean(rayons))
rayons0 = rayons[y == 0] # Cancéreux
print(np.mean(rayons0))
rayons1 = rayons[y == 1]
print(np.mean(rayons1))