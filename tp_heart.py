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

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 500)

df = pd.read_csv("data/heartdisease/data_cleaned_up.csv")
print(df.describe())

df["rnd"] = np.random.rand(len(df))

y = df["num"]
x = df.drop(["num"], axis=1)

np.random.seed(42)

xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, train_size=0.8, test_size=0.2)

scaler = pp.RobustScaler()
scaler.fit(xtrain)
xtrain = scaler.transform(xtrain)
xtest = scaler.transform(xtest)

# model = pipe.make_pipeline(pp.PolynomialFeatures(2), lm.Ridge())
# for k in range(3, 12):
#     model = n.KNeighborsClassifier(n_neighbors=k)
#
#     model.fit(xtrain, ytrain)
#
#     print(k)
#     print(f"Train score: {model.score(xtrain, ytrain)}")
#     print(f"Test score: {model.score(xtest, ytest)}")
model = rf.RandomForestClassifier(n_estimators=100, max_depth=4)
model.fit(xtrain, ytrain)

print(model.feature_importances_)
plt.bar(x.columns, model.feature_importances_)
plt.xticks(rotation=45)
plt.show()

print(f"Train score: {model.score(xtrain, ytrain)}")
print(f"Test score: {model.score(xtest, ytest)}")

values = np.array([[28,1,2,130,132,0,2,185,0,0,0]])
predict = model.predict(values)
print(predict)

tree.export_graphviz(model.estimators_[0], out_file="data/heartdisease/tree.dot", feature_names=x.columns, class_names=["0", "1"])