# Charger avec pandas data/heartdisease/data_cleaned_up.csv
# ok = num == 0
# ko = num == 1

# stats => describe()
# y = dataframe["num"]
# x = dataframe.drop("num", axis=1)
# Créer le modèle LinearRegression
# fit
# predict
# score

import pandas as pd
import sklearn.linear_model as lm


dataframe = pd.read_csv("data/heartdisease/data_cleaned_up.csv")
pd.options.display.max_columns = None
ok = dataframe[dataframe["num"] == 0]
ko = dataframe[dataframe["num"] == 1]

print(ok.describe())
print(ko.describe())

y = dataframe["num"]
x = dataframe.drop("num", axis=1)

model = lm.LinearRegression()
model.fit(x, y)
print(model.score(x, y))
ypred = model.predict(x)

