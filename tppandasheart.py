# Charger le fichier data/heartdisease/data_with_nan.csv pd.read_csv(filename,na_values=".")
# Afficher l'entête du dataframe et noter les valeurs de l'avant dernière colonne
# Effacer les colonnes slope,ca,thal : dataframe.drop(nomcol,1)
# Nettoyer le fichier en remplacant les nan par 0 : fillna
# Calculer la moyenne et l'écart type de chol
# Refaire en effacant les nan : dropna
# ReCalculer la moyenne et l'écart type de chol

import pandas as pd
import numpy as np
