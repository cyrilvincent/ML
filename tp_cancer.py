import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import sklearn.preprocessing as pp
import sklearn.pipeline as pipe
import sklearn.model_selection as ms
import numpy as np
import sklearn.neighbors as n
import sklearn.ensemble as rf
import sklearn.tree as tree
import pickle
import emlearn

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 500)

df = pd.read_csv("data/breast-cancer/data.csv")
print(df.describe())

y = df["diagnosis"]
x = df.drop(["diagnosis", "id"], axis=1)

np.random.seed(42)

xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, train_size=0.8, test_size=0.2)

scaler = pp.RobustScaler()
scaler.fit(xtrain)
xtrain = scaler.transform(xtrain)
xtest = scaler.transform(xtest)

# for k in range(3, 12):
#     model = n.KNeighborsClassifier(n_neighbors=k, )
#     model.fit(xtrain, ytrain)
#
#     print(k)
#     print(f"Train score: {model.score(xtrain, ytrain):.1f}")
#     print(f"Test score: {model.score(xtest, ytest):.1f}")

model = rf.RandomForestClassifier(n_estimators=100, max_depth=5)
model.fit(xtrain, ytrain)

print(model.feature_importances_)
plt.bar(x.columns, model.feature_importances_)
plt.xticks(rotation=45)
plt.show()

tree.export_graphviz(model.estimators_[0], out_file="data/breast-cancer/tree.dot", feature_names=x.columns, class_names=["0", "1"])

print(f"Train score: {model.score(xtrain, ytrain)}")
print(f"Test score: {model.score(xtest, ytest)}")

ypred = model.predict(xtest)
print(ypred)

# scaler.inverse_transform(xtrain)

with open(f"data/breast-cancer/rf-{model.score(xtest, ytest):.2f}.pkl", "wb") as f:
    pickle.dump([scaler, model], f)

cmodel = emlearn.convert(model, method='inline')
cmodel.save(file="data/breast-cancer/rf.h", name='rf')



