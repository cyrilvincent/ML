import pandas as pd
import sklearn.model_selection as ms
import sklearn.neighbors as ne
import numpy as np
import sklearn.ensemble as rf
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
import pickle
import sklearn.svm as svm
import sklearn.neural_network as nn
import sklearn.preprocessing as pp
import sklearn.metrics as metrics

np.random.seed(0)
dataframe = pd.read_csv("data/breast-cancer/data.csv")
y = dataframe.diagnosis
x = dataframe.drop(["diagnosis", "id"], axis=1)



xtrain, xtest, ytrain, ytest = ms.train_test_split(x,y, train_size=0.8, test_size=0.2)
# model = ne.KNeighborsClassifier(n_neighbors=5)
# model = rf.RandomForestClassifier(warm_start=True)
# model = svm.SVC(kernel="linear")

scaler = pp.RobustScaler()
scaler.fit(xtrain)
xtrain = scaler.transform(xtrain)
xtest = scaler.transform(xtest)

model = nn.MLPClassifier(hidden_layer_sizes=(30,20,10) )

model.fit(xtrain, ytrain)
score = model.score(xtest, ytest)
print(score)
ypred = model.predict(xtest)

print(f"Accuracy= {metrics.accuracy_score(ytest, ypred)}")
print(f"Recall= {metrics.recall_score(ytest, ypred)}")
print(f"Precision= {metrics.precision_score(ytest, ypred)}")
print(f"f1_score= {metrics.f1_score(ytest, ypred)}")

print(metrics.classification_report(ytest, ypred))
print(metrics.confusion_matrix(ytest, ypred))

with open("data/breast-cancer/model_rf.pickle", "wb") as f:
    pickle.dump(model, f)

model = None

with open("data/breast-cancer/model_rf.pickle", "rb") as f:
    model = pickle.load(f)

model.fit(x, y)

# print(model.feature_importances_)
# plt.bar(x.columns, model.feature_importances_)
# plt.xticks(rotation=45)
# plt.show()
#
# export_graphviz(model.estimators_[0],
#                 out_file='data/breast-cancer/tree.dot',
#                 feature_names = x.columns,
#                 class_names = str(y),
#                 rounded = True, proportion = False,
#                 precision = 2, filled = True)