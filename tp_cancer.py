import pandas as pd
import numpy as np
import sklearn.model_selection as ms
import sklearn.metrics as metrics
import sklearn.neighbors as nn
import sklearn.preprocessing as pp
import sklearn.ensemble as rf
import matplotlib.pyplot as plt

dataframe = pd.read_csv("data/breast-cancer/data.csv", index_col="id")
y = dataframe.diagnosis
x = dataframe.drop("diagnosis", axis=1)
x["random"] = np.random.rand(len(x))

np.random.seed(0)
xtrain, xtest, ytrain, ytest = ms.train_test_split(x,y,train_size=0.8, test_size=0.2)

scaler = pp.RobustScaler()
scaler.fit(xtrain)
xtrain = scaler.transform(xtrain)
xtest = scaler.transform(xtest)

#model = nn.KNeighborsClassifier(n_neighbors=3)
model = rf.RandomForestClassifier()
model.fit(xtrain, ytrain)
predicted = model.predict(xtest)
print(model.score(xtest, ytest))

plt.bar(x.columns, model.feature_importances_)
plt.xticks(rotation=45)
plt.show()


xsimple = np.array([[17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189,0]])
xsimple = scaler.transform(xsimple)
predicted = model.predict(xsimple)
print(predicted)

from sklearn.tree import export_graphviz

export_graphviz(model.estimators_[0],
                out_file="data/breast-cancer/tree.dot",
                feature_names=x.columns,
                class_names=["0", "1"],
                )

