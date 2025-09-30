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

print(sklearn.__version__)
np.random.seed(42)

#1 Load data
dataframe = pd.read_csv("data/heartdisease/data_cleaned_up.csv")

#2 Make dataset
y = dataframe["num"]
x = dataframe.drop("num", axis=1)

xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, train_size=0.8, test_size=0.2)

scaler = pp.RobustScaler()
scaler.fit(xtrain)
xtrain = scaler.transform(xtrain)
xtest = scaler.transform(xtest)
# scaler.inverse_transform(x)

#5 Creating the model
# model = lm.LinearRegression()
# for k in range(3,12,2):
#     model = n.KNeighborsClassifier(n_neighbors=k)
#     # f(x) = ax + b => 2 poids
#
#     #6 Fit
#     model.fit(xtrain, ytrain)
#
#     #7 Scoring (facultatif)
#     score = model.score(xtest, ytest)
#     print(f"Score: {k} {score:.2f}")

model = rf.RandomForestClassifier(max_depth=5)
model.fit(xtrain, ytrain)



test_score = model.score(xtest, ytest)
train_score = model.score(xtrain, ytrain)
print(train_score, test_score)
with open(f"data/heartdisease/rf-{int(test_score * 100)}.pickle", "wb") as f:
    pickle.dump((scaler, model), f)

from sklearn.tree import export_graphviz
export_graphviz(model.estimators_[0], out_file="data/heartdisease/tree.dot", feature_names=list(x.columns), class_names=["0", "1"])

print(model.feature_importances_)
plt.bar(x.columns, model.feature_importances_)
plt.xticks(rotation=45)
plt.show()