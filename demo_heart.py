import pandas as pd
import numpy as np
import sklearn.model_selection as ms
import sklearn.neighbors as nn
import sklearn.ensemble as tree
import matplotlib.pyplot as plt

np.random.seed(0)


df = pd.read_csv("data/heartdisease/data_cleaned_up.csv")
print(df)
y = df.num
x = df.drop("num", 1)
# ok_set = training_set[df.num == 0]
# ko_set = training_set[df.num == 1]
# print(ok_set.describe().T)
# print(ko_set.describe().T)
# print(df.corr())

np.random.seed(0)
xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, train_size=0.8, test_size=0.2)


# model = lm.LinearRegression()
model = tree.RandomForestClassifier(max_depth=3)
model.fit(xtrain, ytrain)
score = model.score(xtrain, ytrain)
print(score)
score = model.score(xtest, ytest)
print(score)
print(len(ytest))

plt.bar(x.columns, model.feature_importances_)
plt.xticks(rotation=45)
plt.show()

from sklearn.tree import export_graphviz
export_graphviz(model.estimators_[0],
                 out_file='data/heartdisease/tree.dot',
                 feature_names = x.columns,
                 class_names = str(y),
                 rounded = True, proportion = False,
                 precision = 2, filled = True)

