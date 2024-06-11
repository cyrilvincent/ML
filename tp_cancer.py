import pandas as pd
import sklearn.neighbors as neighbors
import sklearn.model_selection as ms
import numpy as np
import sklearn.ensemble as rf
import matplotlib.pyplot as plt

dataframe = pd.read_csv("data/breast-cancer/data.csv", index_col="id")
print(dataframe)
y = dataframe.diagnosis
x = dataframe.drop("diagnosis", axis=1)
np.random.seed(42)

xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, train_size=0.8, test_size=0.2)
print(xtest.shape, ytest.shape)

model = rf.RandomForestClassifier()
model.fit(xtrain, ytrain)
# for k in range(1, 15, 2):
#     model = neighbors.KNeighborsClassifier(n_neighbors=k)
#     model.fit(xtrain, ytrain)
#     print(model.score(xtrain, ytrain), model.score(xtest, ytest))
print(model.score(xtrain, ytrain), model.score(xtest, ytest))
ypred = model.predict(xtest)
print(ypred)
print(model.feature_importances_)

plt.bar(x.columns, model.feature_importances_)
plt.xticks(rotation=45)
plt.show()

from sklearn.tree import export_graphviz
export_graphviz(model.estimators_[0],
                 out_file='data/breast-cancer/tree.dot',
                 feature_names = x.columns,
                 class_names = ["0", "1"],
                 rounded = True, proportion = False,
                 precision = 2, filled = True)
