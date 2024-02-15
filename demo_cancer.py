import pandas as pd
import sklearn.neighbors as nn
import sklearn.model_selection as ms
import sklearn.ensemble as rf
import numpy as np
import matplotlib.pyplot as plt
import pickle

dataframe = pd.read_csv("data/breast-cancer/data.csv", index_col="id")
print(dataframe.describe().T)

y = dataframe.diagnosis
x = dataframe.drop("diagnosis", axis=1)

np.random.seed(0)

xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, train_size=0.8, test_size=0.2)

#model = nn.KNeighborsClassifier(n_neighbors=3)
model = rf.RandomForestClassifier(max_depth=4)
model.fit(xtrain, ytrain)

print(model.score(xtest, ytest), model.score(xtrain, ytrain))

with open("data/breast-cancer/cancer.pickle", "wb") as f:
    pickle.dump(model, f)

model = None

with open("data/breast-cancer/cancer.pickle", "rb") as f:
    model = pickle.load(f)


print(model.feature_importances_)

from sklearn.tree import export_graphviz
export_graphviz(model.estimators_[0],
                 out_file='data/breast-cancer/tree.dot',
                 feature_names = x.columns,
                 class_names = ["0", "1"],
                 rounded = True, proportion = False,
                 precision = 2, filled = True)

plt.bar(x.columns, model.feature_importances_)
plt.xticks(rotation=45)
plt.show()

