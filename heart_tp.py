import pandas as pd
import numpy as np
import sklearn
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
import sklearn.model_selection as ms
import sklearn.neighbors as nn
import sklearn.ensemble as tree

np.random.seed(0)
df = pd.read_csv("data/heartdisease/data_cleaned_up.csv")
y = df.num
x = df.drop("num", axis=1)

xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, train_size=0.8, test_size=0.2)


print(y.describe())
print(x.describe().T)

ok = df[df.num == 0]
ko = df[df.num == 1]
xok = ok.drop("num", axis=1)
xko = ko.drop("num", axis=1)

print(xok.describe().T)
print(xko.describe().T)

# model = lm.LinearRegression()
# model = nn.KNeighborsClassifier(n_neighbors=5)
model = tree.RandomForestClassifier()
model.fit(xtrain, ytrain)
print(model.score(xtrain, ytrain), model.score(xtest, ytest))
y_pred = model.predict(xtest)
print(y_pred)
# print(model.coef_, model.intercept_)
print(model.feature_importances_)

plt.bar(x.columns, model.feature_importances_)
plt.show()

from sklearn.tree import export_graphviz
export_graphviz(model.estimators_[0],
                 out_file='data/heartdisease/tree.dot',
                 feature_names = x.columns,
                 class_names = str(y),
                 rounded = True, proportion = False,
                 precision = 2, filled = True)

# Instancier le LinearModel
# Fit sans values.reshape sur x
# Predict
# afficher coef et intercept
# Score
