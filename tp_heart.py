import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn.pipeline as pipe
import sklearn.preprocessing as pp
import sklearn.model_selection as ms
import sklearn.neighbors as n
import sklearn.ensemble as rf
import sklearn.tree as tree

print("Hello")
print(pd.__version__)

df = pd.read_csv("data/heartdisease/data_cleaned_up.csv")
print(df.describe())

y = df["num"]
x = df.drop(["num"], axis=1)
x["random"] = np.random.rand(len(x))

xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, train_size=0.8, test_size=0.2)

# model = n.KNeighborsClassifier(n_neighbors=3)
model = rf.RandomForestClassifier(n_estimators=100)
model.fit(xtrain, ytrain)

print("Training Score", model.score(xtrain, ytrain))
print("Testing Score", model.score(xtest, ytest))

print(model.feature_importances_)
plt.bar(x.columns, model.feature_importances_)
plt.xticks(rotation=45)
plt.show()

tree.export_graphviz(model.estimators_[0], out_file="data/heartdisease/tree.dot", feature_names=x.columns, class_names=["0", "1"])


