# df = pd.read_csv("data/breast-cancer/data.csv", index_col="id")
# y = diagnosis
# x tout sauf diagnosis
# Humain = 95%

import pandas as pd
import numpy as np
import sklearn.model_selection as ms
import sklearn.neighbors as nn
import sklearn.ensemble as tree
import matplotlib.pyplot as plt
import sklearn.svm as svm
import sklearn.preprocessing as pp
import sklearn.neural_network as neural
import xgboost as xg
import sklearn.metrics as metrics

df = pd.read_csv("data/breast-cancer/data.csv", index_col="id")
np.random.seed(0)
y = df.diagnosis
x = df.drop("diagnosis", axis=1)

xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, train_size=0.8, test_size=0.2)
# model = nn.KNeighborsClassifier(n_neighbors=5)
model = tree.RandomForestClassifier()
model.fit(xtrain, ytrain)
print(model.score(xtrain, ytrain), model.score(xtest, ytest))
ypred = model.predict(xtest)

plt.bar(x.columns, model.feature_importances_)
plt.xticks(rotation=45)
plt.show()

from sklearn.tree import export_graphviz
export_graphviz(model.estimators_[0],
                 out_file='data/breast-cancer/tree.dot',
                 feature_names = x.columns,
                 class_names = ["0", "1"],
                 rounded = True, proportion = False,
                 precision = 2, filled = True)
