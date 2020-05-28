import pandas as pd
import sklearn.preprocessing as pp
import sklearn.pipeline as pipe
import sklearn.model_selection as ms
import sklearn.linear_model as sklm
import matplotlib.pyplot as plt

dataset = pd.read_csv("data/covid-france.txt")
x = dataset.ix[dataset.ix > 50].values.reshape(-1,1)
y = dataset.DC[dataset.ix > 50]

xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, train_size=0.8, test_size=0.2)

model = pipe.make_pipeline(pp.PolynomialFeatures(4), sklm.Ridge())
model.fit(xtrain, ytrain)
print(model.score(xtest, ytest))
plt.bar(x.reshape(-1), y)
plt.plot(x.reshape(-1), model.predict(x).reshape(-1))
plt.show()

