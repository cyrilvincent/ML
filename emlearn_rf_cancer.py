import pandas as pd
import numpy as np
import sklearn.model_selection as ms
import sklearn.ensemble as rf
import sklearn.preprocessing as pp
import emlearn
import matplotlib.pyplot as plt

np.random.seed(42)

pd.set_option('display.max_columns', None)
dataframe = pd.read_csv("data/breast-cancer/data.csv")

y = dataframe["diagnosis"]
x = dataframe.drop(["diagnosis", "id"], axis=1)

xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, train_size=0.8, test_size=0.2,)

scaler = pp.StandardScaler()
scaler.fit(xtrain)
xtrain = scaler.transform(xtrain)
xtest = scaler.transform(xtest)

model = rf.RandomForestClassifier(n_estimators=20, max_depth=6, max_features=2, random_state=1)
model.fit(xtrain, ytrain)
score_train = model.score(xtrain, ytrain)
score_test = model.score(xtest, ytest)
print(score_train, score_test)

cmodel = emlearn.convert(model, method='inline')
cmodel.save(file="data/breast-cancer/cancer_rf_model.csv", name='rf', format='csv')

# tree.export_graphviz(model.estimators_[0], out_file="data/breast-cancer/tree.dot", feature_names=x.columns, class_names=["0", "1"])
#
print(model.feature_importances_)
print(np.argsort(model.feature_importances_)[-5:][::-1])
plt.bar(x.columns, model.feature_importances_)
plt.xticks(rotation=45)
plt.show()




