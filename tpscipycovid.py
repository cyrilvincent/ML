# Installer scipy
# Calculer la régression linéaire surfaces / loyers
# surfaces = surfaces[surfaces < 200] recalcul
# Charger covid-france.txt er créer les tableaux nbcas, dcs
# Afficher dans un graphique en barre (bar)
# mean, max, min, std
# Calculer la régression linéaire et voir qu'elle ne marche pas
# Facultatif : calculer la régression linéaire sur ix < 54

import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
import csv

with open("data/covid-france.txt") as f:
    reader = csv.DictReader(f)
    ix = []
    nbcas = []
    dcs = []
    for row in reader:
        ix.append(float(row["ix"]))
        nbcas.append(float(row["NbCas"]))
        dcs.append(float(row["DC"]))
    ix = np.array(ix)
    nbcas = np.array(nbcas)
    dcs = np.array(dcs)

    # nbcas = nbcas[ix < 54]
    # ix = ix[ix < 54]

slope, intercept, r_value, p_value, std_err = stats.linregress(ix, nbcas)
print(slope, intercept, r_value, p_value, std_err)

f = lambda x : slope * x + intercept
plt.bar(ix, nbcas)

plt.show()
plt.bar(ix, dcs)
plt.show()


