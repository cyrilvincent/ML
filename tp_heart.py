# Charger avec pandas data/heartdisease/data_cleaned_up.csv
# ok = num == 0
# ko = num == 1
# stats => describe()
y = dataframe["num"]
x = dataframe.drop("num", axis=1)
# Créer le modèle LinearRegression
# fit
# predict
# score
