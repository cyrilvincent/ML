import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import sklearn.linear_model as lm
import sklearn.pipeline as pipe
import sklearn.preprocessing as pp
import sklearn.model_selection as ms


dataframe = pd.read_csv("data/house/house.csv")
dataframe = dataframe[dataframe["surface"]<200]
print(dataframe.describe())

slope, intercept, rvalue, pvalue, mse = stats.linregress(dataframe["surface"], dataframe["loyer"])
print(slope, intercept, rvalue, pvalue, mse)

x = np.arange(200)
y = slope * x + intercept
plt.plot(x, y, color="red")


x = dataframe["surface"].values.reshape(-1,1)
y = dataframe["loyer"]

xtrain, xtest, ytrain, ytest = ms.train_test_split(x,y,train_size=0.8,test_size=0.2)



model = lm.LinearRegression()
model.fit(xtrain, ytrain)
ypred = model.predict(xtest)
print(model.coef_, model.intercept_) # Variables privées non portables entre modèles
score_train = model.score(xtrain,ytrain)
print(f"Train score {score_train}")
score_test = model.score(xtest,ytest)
print(f"Test score {score_test}")
plt.plot(xtest.reshape(-1, 1), ypred, color="green")

model = pipe.make_pipeline(pp.PolynomialFeatures(3), lm.Ridge())
model.fit(x, y)
print(model.score(x, y))
ypred = model.predict(np.arange(200).reshape(-1,1))
plt.plot(np.arange(200).reshape(-1,1), ypred, color="blue")

plt.scatter(dataframe["surface"], dataframe["loyer"])
plt.show()
