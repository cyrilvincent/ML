# Charger le fichier heartdisease/data_cleaned_up.csv
# Describe
# ok = dataset où num == 0
# ko = dataset où num == 1
# par 2 simples describe sur ok et ko trouver des corrélations
# sur le dataframe initial tester la méthode .corr()

import pandas as pd
dataframe = pd.read_csv("data/heartdisease/data_cleaned_up.csv")
ok = dataframe[dataframe.num == 0]
ko = dataframe[dataframe.num == 1]

print(ok.describe().T)
print(ko.describe().T)

print(dataframe.corr())