import pandas as pd
import numpy as np
import sklearn.model_selection as ms
import sklearn.neighbors as nn
import sklearn.ensemble as rf
import sklearn.svm as svm
import matplotlib.pyplot as plt
import pickle
import sklearn.preprocessing as pp
import sklearn.neural_network as neural

dataframe = pd.read_csv("data/breast-cancer/data.csv", index_col='id')
print(dataframe)
y = dataframe.diagnosis
x = dataframe.drop("diagnosis", 1)

# Fit
# Score = accuracy = nbtrue / len
# Predict KNN

np.random.seed(0)
xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, train_size=0.8, test_size=0.2)

scaler = pp.RobustScaler()
scaler.fit(xtrain)
xtrain = scaler.transform(xtrain)
xtest = scaler.transform(xtest)
# model = nn.KNeighborsClassifier(n_neighbors=7)
# model = rf.RandomForestClassifier(warm_start=True)
model = neural.MLPClassifier((30,10), max_iter=1000)
# model = svm.SVC(C=1, kernel="poly", degree=2)
model.fit(xtrain, ytrain)
score = model.score(xtest, ytest)
print(score)

# with open(f"data/breast-cancer/rf_model_{score:.3f}.pickle", "wb") as f:
#     pickle.dump(model, f)
#
# model = None
#
# with open(f"data/breast-cancer/rf_model_0.974.pickle", "rb") as f:
#     model = pickle.load(f)

# ypredict = model.predict(xtest)
# errors = ypredict - ytest
# print(errors.values)
# score = 1 - len(errors[np.abs(errors) == 1]) / len(ytest)
# print(score)
# fp = 1 - len(errors[errors == 1]) / len(ytest)
# print(fp)

# plt.bar(xtest.columns, model.feature_importances_)
# plt.xticks(rotation = 90)
# plt.show()


