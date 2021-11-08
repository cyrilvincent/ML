import pandas as pd
import numpy as np

dataframe = pd.read_csv("data/breast-cancer/data.csv", index_col="id")
print(dataframe)

# Afficher le describe
# Afficher diagnosis, radius_mean, concavity_se
# Filtrer les patients sains et malades
# Afficher la moyenne (np.mean()) de radius_mean, concavity_se pour les 2 groupes de patients
sains = dataframe[dataframe.diagnosis == 0]