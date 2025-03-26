import pandas as pd
import numpy as np
import sklearn.neighbors as n
import sklearn.model_selection as ms
import sklearn.preprocessing as pp
import sklearn.ensemble as rf
import matplotlib.pyplot as plt
import pickle
import sklearn.neural_network as nn

dataframe = pd.read_csv("data/breast-cancer/data.csv", index_col="id")
y = dataframe.diagnosis
dataframe["rnd"] = np.random.rand()
x = dataframe.drop("diagnosis", axis=1)

xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, test_size=0.2, train_size=0.8)

# KNN avec k=3
# tester avec différentes valeurs de k
# RobustScaler
# scaler = model

scaler = pp.RobustScaler()
scaler.fit(xtrain)
xtrain = scaler.transform(xtrain)
xtest = scaler.transform(xtest)

# for k in range(3,15,2):
#     model = n.KNeighborsClassifier(n_neighbors=k)
#
#     model.fit(xtrain, ytrain)
#     score = model.score(xtest, ytest)
#     print(k, score)

# model = rf.RandomForestClassifier()
model =nn.MLPClassifier(hidden_layer_sizes=(30,30,30))
model.fit(xtrain, ytrain)
score = model.score(xtest, ytest)
print(score)
# with open(f"data/breast-cancer/rf-{score:.2f}.pickle", "wb") as f:
#     pickle.dump((model, scaler), f)
#
# from sklearn.tree import export_graphviz
#
# export_graphviz(model.estimators_[0], out_file="data/breast-cancer/tree.dot", feature_names=x.columns, class_names=["0", "1"])
#
# print(model.feature_importances_)
#
# plt.bar(x.columns, model.feature_importances_)
# plt.xticks(rotation=45)
# plt.show()


