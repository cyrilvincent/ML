# Refaire fonctionner dom_sklearn + tp_heart_sklearn
# Refaire tp_heart_sklearn pour le cancer
# y = diagnosis
# x = tout sauf diagnosis et id

# Refaire une regression

import numpy as np
import sklearn
import sklearn.linear_model as lm
import sklearn.preprocessing as pp
import sklearn.pipeline as pipe
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.neighbors as nb
import sklearn.model_selection as ms
import sklearn.ensemble as rf
import pickle
import sklearn.svm as svm
print(sklearn.__version__)

# 1 Pandas DataMart
dataframe = pd.read_csv("data/breast-cancer/data.csv")

# Déterminer x et y
y = dataframe["diagnosis"]
x = dataframe.drop(["diagnosis", "id"], axis=1)
xoriginal = x

scaler = pp.RobustScaler() # Calcul median, les quartiles
scaler.fit(x) # (x - median) / (if x < median => 1/4tile sinon max - 3/4ile)
x=scaler.transform(x)

np.random.seed(42)
xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, train_size=0.8, test_size=0.2)

# x,y = scaler.fit_transform(x, y)

# 3 Model
# model = lm.LinearRegression()
# f(x) = ax + b; a = slope; b = intersection pour x=0;
# a et b soit les monômes du polynôme de dégré 1; a et b sont des poids
# for degree in range(2,5):
degree = 2
# model = pipe.make_pipeline(pp.PolynomialFeatures(degree), lm.Ridge())
# f(x) = ax² + bx + c
# model = nb.KNeighborsClassifier(n_neighbors=3)
# model = rf.RandomForestClassifier(max_depth=4)
model = svm.SVC(kernel="linear")  # kernel = rbf, poly, linear, degree=3

# 4 Apprentissage supervisé car y est connu
model.fit(xtrain, ytrain)

# 5 Score
train_score = model.score(xtrain, ytrain)
test_score = model.score(xtest, ytest)
print("Score", train_score, test_score)

from sklearn.tree import export_graphviz
export_graphviz(model.estimators_[0], out_file="data/breast-cancer/tree.dot", feature_names=xoriginal.columns, class_names=["0", "1"])

print(model.feature_importances_)
plt.bar(xoriginal.columns, model.feature_importances_)
plt.xticks(rotation=45)
plt.show()

with open("data/breast-cancer/rf.pickle", "wb") as f:
    pickle.dump(model, f)

# Sauvegarder le model
# Faire une prédiction dans un nouveau script
# demo mnist

