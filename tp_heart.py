import sklearn.linear_model as lm
import pandas as pd
import numpy as np
import sklearn.neighbors as n
import sklearn.model_selection as ms
import sklearn.ensemble as rf
import sklearn.tree as tree
import matplotlib.pyplot as plt
np.random.seed(42)

pd.set_option('display.max_columns', None)
dataframe = pd.read_csv("data/heartdisease/data_cleaned_up.csv")

print(dataframe.describe())

y = dataframe["num"]
x = dataframe.drop(["num"], axis=1)

xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, train_size=0.8, test_size=0.2,)

# model = n.KNeighborsClassifier(n_neighbors=3)
model = rf.RandomForestClassifier(max_depth=4)

model.fit(xtrain, ytrain)
score_train = model.score(xtrain, ytrain)
score_test = model.score(xtest, ytest)
print(score_train, score_test)

tree.export_graphviz(model.estimators_[0], out_file="data/breast-cancer/tree.dot", feature_names=x.columns, class_names=["0", "1"])

print(model.feature_importances_)
plt.bar(x.columns, model.feature_importances_)
plt.xticks(rotation=45)
plt.show()