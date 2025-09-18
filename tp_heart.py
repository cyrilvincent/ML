# Charger le fichier data_cleaned_up.csv avec pandas
# Faire un describe

import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
dataframe=pd.read_csv("data/heartdisease/data_cleaned_up.csv")
print(dataframe.describe().T)


# ok = num==0
# ko = num==1
# Faire un describe sur les 3 dataframes => en tirer des hypothèses
ok = dataframe[dataframe["num"] == 0]
ko = dataframe[dataframe["num"] == 1]

print(ok.describe().T)
print(ko.describe().T)

print(dataframe.corr())



