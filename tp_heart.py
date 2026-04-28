import numpy as np
import pandas as pd
import sklearn.linear_model as lm
import sklearn.preprocessing as pp
import sklearn.pipeline as pipe
import sklearn.model_selection as ms
import sklearn.neighbors as n

df = pd.read_csv("data/heartdisease/data_cleaned_up.csv")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

y = df["num"]
x = df.drop(["num"], axis=1)

print(df.corr())

np.random.seed(42)
xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, train_size=0.8, test_size=0.2)

# model = lm.LinearRegression()
for k in range(3, 12, 2):
    model = n.KNeighborsClassifier(n_neighbors=k)

    model.fit(xtrain, ytrain)

    print(model.score(xtest, ytest))
# train_test_split
# fit
# predict
# score