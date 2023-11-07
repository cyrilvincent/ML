import pandas as pd
import sklearn.linear_model as lm
import sklearn.neighbors as nn
import sklearn.model_selection as ms
import numpy as np
import sklearn.ensemble as rf
import matplotlib.pyplot as plt

np.random.seed(0)
dataframe = pd.read_csv("data/breast-cancer/data.csv", index_col="id")
y = dataframe.diagnosis
x = dataframe.drop("diagnosis", axis=1)
xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, train_size=0.8, test_size=0.2)
# model = lm.LinearRegression()
# model = nn.KNeighborsClassifier(n_neighbors=3)
model = rf.RandomForestClassifier(max_depth=4)
model.fit(xtrain, ytrain)
print(model.score(xtrain, ytrain))
print(model.score(xtest, ytest))

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
