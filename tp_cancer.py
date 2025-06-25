import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn.model_selection as ms
import sklearn.ensemble as rf
from sklearn.tree import export_graphviz
import sklearn.preprocessing as pp
import pickle
import sklearn.neural_network as neural

dataframe = pd.read_csv("data/breast-cancer/data.csv")

y = dataframe.diagnosis
dataframe["rnd"] = np.random.rand(y.shape[0])
x = dataframe.drop(["id", "diagnosis"], axis=1)

np.random.seed(42)
xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, train_size=0.8, test_size=0.2)

scaler = pp.RobustScaler()
scaler.fit(xtrain)
xtrain = scaler.transform(xtrain)
xtest = scaler.transform(xtest)

# model = rf.RandomForestClassifier()
model = neural.MLPClassifier(hidden_layer_sizes=(30,30,30))
model.fit(xtrain, ytrain)
score_train = model.score(xtrain, ytrain)
score_test = model.score(xtest, ytest)
print(f"Train score: {score_train:.2f}")
print(f"Test score: {score_test:.2f}")

with open(f"data/breast-cancer/mlp-{int(score_test * 100)}.pickle", "wb") as f:
    pickle.dump((model, scaler), f)


# export_graphviz(model.estimators_[0], out_file="data/breast-cancer/tree.dot", feature_names=x.columns, class_names=["0", "1"])
#
# print(model.feature_importances_)
#
# plt.bar(x.columns, model.feature_importances_)
# plt.xticks(rotation=45)
# plt.show()


