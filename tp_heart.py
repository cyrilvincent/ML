import pandas as pd
import sklearn.linear_model as lm
import numpy as np
import sklearn.neighbors as n
import sklearn.model_selection as ms
import sklearn.ensemble as rf
import matplotlib.pyplot as plt
import numpy as np



dataframe = pd.read_csv("data/heartdisease/data_cleaned_up.csv")
dataframe["rnd"] = np.random.rand()

y = dataframe.num
x = dataframe.drop("num", axis=1)

xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, test_size=0.2, train_size=0.8)

# model = lm.LinearRegression()
# model = n.KNeighborsClassifier(n_neighbors=3)
model = rf.RandomForestClassifier()

model.fit(xtrain, ytrain)

score = model.score(xtrain, ytrain)
print(score)
score = model.score(xtest, ytest)
print(score)

predicted = model.predict(xtest)
print(predicted)

from sklearn.tree import export_graphviz

export_graphviz(model.estimators_[0], out_file="data/heartdisease/tree.dot", feature_names=x.columns, class_names=["0", "1"])

print(model.feature_importances_)

plt.bar(x.columns, model.feature_importances_)
plt.xticks(rotation=45)
plt.show()