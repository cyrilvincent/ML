import sklearn
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import sklearn.pipeline as pipe
import sklearn.preprocessing as pp
import sklearn.model_selection as ms
import sklearn.neighbors as n
import numpy as np
import sklearn.ensemble as rf
import sklearn.tree as tree

df = pd.read_csv("data/heartdisease/data_cleaned_up.csv")
print(df.describe())

np.random.seed(42)

y = df["num"]
x = df.drop(["num"], axis=1)

xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, train_size=0.8, test_size=0.2)

print(xtest.shape)

k=3
# model = n.KNeighborsClassifier(n_neighbors=k)
model = rf.RandomForestClassifier()
model.fit(xtrain, ytrain)
print(f"Train score: {model.score(xtrain, ytrain):.2f}")
print(f"Test score: {model.score(xtest, ytest):.2f}")

tree.export_graphviz(model.estimators_[0], out_file="data/heartdisease/tree.dot", feature_names=xtrain.columns, class_names=["0","1"])

print(model.feature_importances_)

plt.bar(x.columns, model.feature_importances_)
plt.show()

