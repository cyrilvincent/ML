import pandas as pd
import sklearn.linear_model as lm
import sklearn.neighbors as neighbors
import sklearn.model_selection as ms

pd.options.display.width = 0
dataframe = pd.read_csv("data/heartdisease/data_cleaned_up.csv")

ok = dataframe[dataframe.num == 0]
ko = dataframe[dataframe.num == 1]
print(ok.chol.describe())
print(ko.chol.describe())

print(dataframe.corr())

y = dataframe.num
x = dataframe.drop("num", axis=1)
xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, train_size=0.8, test_size=0.2)

#model = lm.LinearRegression()
model = neighbors.KNeighborsClassifier(n_neighbors=3)
model.fit(xtrain, ytrain)
ypred = model.predict(xtest)
score = model.score(xtest, ytest)
print(score)
score = model.score(xtrain, ytrain)
print(score)
