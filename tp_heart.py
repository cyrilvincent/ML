# Charger avec pandas data/heartdisease/data_cleaned_up.csv
# ok = num == 0
# ko = num == 1

# stats => describe()
# y = dataframe["num"]
# x = dataframe.drop("num", axis=1)
# Créer le modèle LinearRegression
# fit
# predict
# score

import pandas as pd
import sklearn.linear_model as lm
import sklearn.model_selection as ms
import numpy as np
import sklearn.neighbors as nn

np.random.seed(42)


dataframe = pd.read_csv("data/heartdisease/data_cleaned_up.csv")
pd.options.display.max_columns = None
ok = dataframe[dataframe["num"] == 0]
ko = dataframe[dataframe["num"] == 1]

print(ok.describe())
print(ko.describe())

y = dataframe["num"]
x = dataframe.drop("num", axis=1)
xtrain, xtest,ytrain,ytest = ms.train_test_split(x,y,train_size=0.8,test_size=0.2)

# model = lm.LinearRegression()
model = nn.KNeighborsClassifier(n_neighbors=3)
model.fit(xtrain, ytrain)
print(model.score(xtest, ytest))
ypred = model.predict(xtest)




