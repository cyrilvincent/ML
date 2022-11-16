# Dans heartdisease data_with_nan.csv & data_cleaned_up.csv
# Le contenu du fichier est expliqué dans index.txt
# Essayer de comprendre le fichier
# Essayer de faire les stats appropriées sur data_cleaned_up.csv

import pandas as pd
import sklearn.model_selection as ms
import sklearn.neighbors as ne
import numpy as np
import sklearn.ensemble as rf

np.random.seed(0)
dataframe = pd.read_csv("data/heartdisease/data_cleaned_up.csv")
y = dataframe.num
x = dataframe.drop("num", axis=1)

xtrain, xtest, ytrain, ytest = ms.train_test_split(x,y, train_size=0.8, test_size=0.2)
# model = ne.KNeighborsClassifier(n_neighbors=5)
model = rf.RandomForestClassifier()
model.fit(xtrain, ytrain)
score = model.score(xtest, ytest)
print(score)

data = [28,1,2,130.0,132.0,0.0,2.0,185.0,0.0,0.0]
ypred = model.predict([data])
print(ypred)


