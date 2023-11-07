import pandas as pd
import sklearn.neighbors as nn
import sklearn.model_selection as ms

dataframe = pd.read_csv("data/heartdisease/data_cleaned_up.csv")
y = dataframe.num
x = dataframe.drop("num", axis=1)

ok = dataframe[dataframe.num == 0]
ko = dataframe[dataframe.num == 1]

print(ok.describe().T)
print(ko.describe().T)

xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, train_size=0.8, test_size=0.2)
# model = lm.LinearRegression()
model = nn.KNeighborsClassifier(n_neighbors=3)
model.fit(xtrain, ytrain)
print(model.score(xtrain, ytrain))
print(model.score(xtest, ytest))