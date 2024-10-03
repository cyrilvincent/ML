# Charger data/breast-cancer/data.csv
# pd.read_csv("data/breast-cancer/data.csv", index_col="id")
# seed + train test + knn + fit + score + predict + conclusion
# tester différentes valeurs de k

import pandas as pd
import numpy as np
import sklearn.neighbors as nn
import sklearn.model_selection as ms
import sklearn.preprocessing as pp
import sklearn.ensemble as rf
import sweetviz
import matplotlib.pyplot as plt
import pickle
import sklearn.svm as svm

dataframe = pd.read_csv("data/breast-cancer/data.csv", index_col="id")
print(dataframe.describe())
# report = sweetviz.analyze(dataframe)
# report.show_html("data/breast-cancer/report.html")

np.random.seed(42)

y = dataframe["diagnosis"]
x = dataframe.drop("diagnosis", axis=1)

xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, train_size=0.8, test_size=0.2)

scaler = pp.RobustScaler()
scaler.fit(xtrain)
xtrain = scaler.transform(xtrain)
xtest = scaler.transform(xtest)

# for k in range(3,12,2):
#     model = nn.KNeighborsClassifier(n_neighbors=k)
#
#     model.fit(xtrain, ytrain)
#
#     scoretrain = model.score(xtrain, ytrain)
#     scoretest = model.score(xtest, ytest)
#
#     print(scoretrain, scoretest)

model = rf.RandomForestClassifier(max_leaf_nodes=20)
model = svm.SVC(C=0.1)
model.fit(xtrain, ytrain)
scoretrain = model.score(xtrain, ytrain)
scoretest = model.score(xtest, ytest)
print(scoretrain, scoretest)
with open(f"data/breast-cancer/rf-{int(scoretest*100)}.pickle", "wb") as f:
    pickle.dump((scaler, model), f)

ypred = model.predict(xtest)
print(ypred)
plt.bar(x.columns, model.feature_importances_)
plt.xticks(rotation=45)
plt.show()

from sklearn.tree import export_graphviz
export_graphviz(model.estimators_[0],
                out_file="data/breast-cancer/tree.dot",
                feature_names=x.columns,
                class_names=["0", "1"],
                rounded=True, proportion=False, precision=2, filled=True)


