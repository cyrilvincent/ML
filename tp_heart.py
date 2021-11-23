# Charger data/heartdisease/data_with_nan.csv
# Dropper les colonnes slope,ca,thal
# Dropper les lignes avec des nan : dropna()
# Sauvegarder dans dataclean.csv : to_csv
# Créer 2 datasets : ok pour les patients sains : num = 0
#                    ko pour les malades : num = 1
# Afficher les moyennes et écart types (mean, std) des 2 datasets pour la colonne chol
# Afficher les corrélations des datasets : corr()

import pandas as pd
import numpy as np
import sklearn.linear_model as lm
import sklearn.ensemble as rf
import sklearn.tree as tree
import matplotlib.pyplot as plt



dataframe = pd.read_csv("data/heartdisease/data_with_nan.csv", na_values='.')
dataframe = dataframe.drop("slope", axis=1).drop("ca", 1).drop("thal", 1)
dataframe = dataframe.dropna()
dataframe.to_csv("data/heartdisease/dataclean.csv")
ok = dataframe[dataframe.num == 0]
ko = dataframe[dataframe.num == 1]
print(np.mean(ok.chol), np.std(ok.chol))
print(np.mean(ko.chol), np.std(ko.chol))
print(ko.corr())

y = dataframe.num
x = dataframe.drop("num", 1)

print(x.shape)

# Instanciation du model
# model = lm.LinearRegression()
model = rf.RandomForestClassifier()

# Apprentissage
model.fit(x, y)
# print(model.coef_, model.intercept_)
# Prediction
score = model.score(x, y)
print(score)

print(model.feature_importances_)
plt.bar(x.columns, model.feature_importances_)
plt.show()


tree.export_graphviz(model.estimators_[0], out_file="data/heartdisease/tree.dot", feature_names=x.columns,
                     class_names=["0", "1"], filled=True)
