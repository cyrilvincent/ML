# Avec pandas load data/breast-cancer/data.Csv
# y = diagnosis
# x tout sauf diagnosis et id
import sklearn
import pandas as pd
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
import numpy as np
import sklearn.neighbors as n
import sklearn.model_selection as ms
import sklearn.preprocessing as pp
import sklearn.ensemble as rf
import pickle
import sklearn.neural_network as nn
import sklearn.metrics as metrics

np.random.seed(42)

#1 Load data
dataframe = pd.read_csv("data/breast-cancer/data.csv")

#2 Make dataset
y = dataframe["diagnosis"]
x = dataframe.drop(["diagnosis", "id"], axis=1)

xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, train_size=0.8, test_size=0.2)

scaler = pp.RobustScaler()
scaler.fit(xtrain)
xtrain = scaler.transform(xtrain)
xtest = scaler.transform(xtest)

#5 Creating the model
# model = lm.LinearRegression()
# model = n.KNeighborsClassifier(n_neighbors=11)
# f(x) = ax + b => 2 poids
# model = rf.RandomForestClassifier(max_depth=5)
model = nn.MLPClassifier(hidden_layer_sizes=(20,), max_iter=1000)
model.fit(xtrain, ytrain)
test_score = model.score(xtest, ytest)
train_score = model.score(xtrain, ytrain)
print(train_score, test_score)
with open(f"data/breast-cancer/rf-{int(test_score * 100)}.pickle", "wb") as f:
    pickle.dump((scaler, model), f)

ypredicted = model.predict(xtest)
print(metrics.classification_report(y_true=ytest, y_pred=ypredicted))
print(metrics.confusion_matrix(y_true=ytest, y_pred=ypredicted))

# from sklearn.tree import export_graphviz
# export_graphviz(model.estimators_[0], out_file="data/breast-cancer/tree.dot", feature_names=list(x.columns), class_names=["0", "1"])
#
# print(model.feature_importances_)
# plt.bar(x.columns, model.feature_importances_)
# plt.xticks(rotation=45)
# plt.show()