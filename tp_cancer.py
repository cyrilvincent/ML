import pandas as pd
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
import sklearn.pipeline as pipe
import sklearn.preprocessing as pre
import sklearn.model_selection as ms
import sklearn.neighbors as n
import sklearn.preprocessing as pp
import numpy as np
import sklearn.ensemble as rf
import sklearn.tree as tree

df = pd.read_csv("data/breast-cancer/data.csv")

df["rnd"] = np.random.rand(len(df))
y = df["diagnosis"]
x = df.drop(["diagnosis", "id"], axis=1)
columns = x.columns

np.random.seed(42)

xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, train_size=0.8, test_size=0.2)

scaler = pp.RobustScaler()
scaler.fit(xtrain)
xtrain = scaler.transform(xtrain)
xtest = scaler.transform(xtest)

# model = n.KNeighborsClassifier(n_neighbors=3)
model = rf.RandomForestClassifier(max_depth=5)
model.fit(xtrain, ytrain)

ypred = model.predict(xtest)

trainscore = model.score(xtrain, ytrain)
testscore = model.score(xtest, ytest)

print(trainscore, testscore)

tree.export_graphviz(model.estimators_[0], "data/breast-cancer/tree.dot",class_names=["0", "1"], feature_names= columns )
print(model.feature_importances_)

plt.bar(columns, model.feature_importances_)
plt.xticks(rotation=45)
plt.show()
