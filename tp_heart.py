import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import sklearn.preprocessing as pp
import sklearn.pipeline as pipe
import sklearn.model_selection as ms

dataframe = pd.read_csv("data/heartdisease/data_cleaned_up.csv")
# y = dataframe["num"]
# x = dataframe.drop("num", 1)

plt.matshow(dataframe.corr())
plt.show()

ok = dataframe[dataframe["num"] == 0]
ko = dataframe[dataframe["num"] == 1]

print(f"OK chol mean: {np.mean(ok.chol)}, std: {np.std(ok.chol)}")
print(f"KO chol mean: {np.mean(ko.chol)}, std: {np.std(ko.chol)}")

print(f"OK sex mean: {np.mean(ok.sex)}, std: {np.std(ok.sex)}")
print(f"KO sex mean: {np.mean(ko.sex)}, std: {np.std(ko.sex)}")

print(f"OK age mean: {np.mean(ok.age)}, std: {np.std(ok.age)}")
print(f"KO age mean: {np.mean(ko.age)}, std: {np.std(ko.age)}")
print(f"Dataframe age mean: {np.mean(dataframe.age)}, std: {np.std(dataframe.age)}")

# afficher les correlations sur le dataframe
# ok = filtrer sur num == 0
# ko = filtrer sur num == 1
# sur chol calculer mean, std pour ok et ko => conclusion
# idem sur sex et age

y = dataframe["num"]
x = dataframe.drop("num", axis=1)

xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, train_size=0.8, test_size=0.2)

# model = lm.LinearRegression()
model = pipe.make_pipeline(pp.PolynomialFeatures(2), lm.Ridge())
model.fit(xtrain, ytrain)
ypredicted = model.predict(xtest)
print(model.score(xtest, ytest))
# Créer le modèle à partir du dataframe
# fit
# predict
# score : conclusion

