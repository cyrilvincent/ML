import pandas as pd
import numpy as np
import sklearn.model_selection as ms
import sklearn.neighbors as nn
import sklearn.ensemble as tree
import matplotlib.pyplot as plt
import sklearn.svm as svm
import sklearn.preprocessing as pp

df = pd.read_csv("data/breast-cancer/data.csv", index_col="id")
np.random.seed(0)
y = df.diagnosis
x = df.drop("diagnosis", 1)

xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, train_size=0.8, test_size=0.2)
# model = nn.KNeighborsClassifier(n_neighbors=5)
scaler = pp.RobustScaler()
scaler.fit(xtrain)
xtrain = scaler.transform(xtrain)
xtest = scaler.transform(xtest)
model = tree.RandomForestClassifier(n_estimators=100, max_depth=20)
# model = svm.SVC(kernel="poly", degree=3)
model.fit(xtrain, ytrain)
score = model.score(xtest, ytest)
print(score)
ypred = model.predict(xtest)
print((ypred - ytest).values)

print(model.feature_importances_)
plt.bar(x.columns, model.feature_importances_)
plt.xticks(rotation=45)
plt.show()
print(len(ytest))

from sklearn.tree import export_graphviz
export_graphviz(model.estimators_[0],
                 out_file='data/breast-cancer/tree.dot',
                 feature_names = x.columns,
                 class_names = str(y),
                 rounded = True, proportion = False,
                 precision = 2, filled = True)
