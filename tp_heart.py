# Dans heartdisease data_with_nan.csv & data_cleaned_up.csv
# Le contenu du fichier est expliqué dans index.txt
# Essayer de comprendre le fichier
# Essayer de faire les stats appropriées sur data_cleaned_up.csv

import pandas as pd

dataframe = pd.read_csv("data/heartdisease/dataclean.csv")
ok = dataframe[dataframe.num == 0]
ko = dataframe[dataframe.num == 1]
print("OK")
print(ok.describe().T)
print()
print()
print(ko.describe().T)