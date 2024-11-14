# Pandas charger data/heartdisease/data_cleaned_up.csv
# Analyse statistique
# ok = num == 0
# ko = num == 1
# Faire des stats => describe()
# y = colonne num
# x toutes les autres colonnes : dataframe.drop("num", axis=1)
# Créer le modèle LinearRegression
# Fit
# Predict
# Score => non fonctionnel

import pandas as pd
import sklearn.linear_model as lm

pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import numpy as np
import sklearn.preprocessing as pp
import sklearn.pipeline as pipe
import sklearn.model_selection as ms


dataframe = pd.read_csv("data/heartdisease/data_cleaned_up.csv")
ok = dataframe[dataframe.num == 0]
ko = dataframe[dataframe.num == 1]

y = dataframe.num
x = dataframe.drop("num", axis=1)

np.random.seed(42)
xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, train_size=0.8, test_size=0.2)

# model = lm.LinearRegression()
model = pipe.make_pipeline(pp.PolynomialFeatures(3), lm.Ridge())
model.fit(xtrain, ytrain)

predicted = model.predict(xtest)
score1 = model.score(xtest, ytest)
score2 = model.score(xtrain, ytrain)

print(ok.describe())
print(ko.describe())
print(score1, score2)

# print(np.abs(dataframe.corr()))
