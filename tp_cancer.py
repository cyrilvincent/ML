import pandas as pd
# pip install matplotlib
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import sklearn.preprocessing as pp
import sklearn.pipeline as pipe
import sklearn.neighbors as n
import numpy as np
import sklearn.model_selection as ms
import sklearn.ensemble as rf
import sklearn.tree as tree


df = pd.read_csv("data/breast-cancer/data.csv")

df["rnd"] = np.random.rand()

df.hist(bins=10)
plt.show()
df["concave_points_worst"].hist(bins=20)
plt.show()

y = df["diagnosis"]
x = df.drop(["diagnosis", "id"], axis=1)


np.random.seed(42)
xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, train_size=0.8, test_size=0.2)

scaler = pp.RobustScaler()
scaler.fit(xtrain)
xtrain = scaler.transform(xtrain)
xtest = scaler.transform(xtest)

# for k in range(3, 15, 2):
# model = n.KNeighborsClassifier(n_neighbors=k)
model = rf.RandomForestClassifier(max_depth=6)
model.fit(xtrain, ytrain)

train_score = model.score(xtrain, ytrain)
test_score = model.score(xtest, ytest)

print(train_score, test_score)

print(model.feature_importances_)

plt.bar(x.columns, model.feature_importances_)
plt.xticks(rotation=45)
plt.show()

tree.export_graphviz(model.estimators_[0], out_file="data/breast-cancer/tree.dot", feature_names=x.columns, class_names=["0","1"])
# Graphviz Viewer

ypred = model.predict(xtest)

