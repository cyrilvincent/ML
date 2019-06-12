from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer() # more info : https://goo.gl/U2Uwz2

#input
X=cancer['data']
y=cancer['target']

print(X.shape) #569 * 30
print(y.shape) #569

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y)

import sklearn.ensemble as rf
model = rf.RandomForestClassifier(n_estimators=100, warm_start=True)
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print(score)

predicted = model.predict(X_test)
print(predicted)
print(y_test)
print(predicted - y_test)

features_importances = model.feature_importances_
print(features_importances)

import matplotlib.pyplot as plt
plt.bar(range(len(features_importances)),features_importances)
plt.show()

import pickle
with open("breastcancer/cancer-rf.pickle","wb") as f:
    pickle.dump(model, f)

model = None

with open("breastcancer/cancer-rf.pickle","rb") as f:
    model = pickle.load(f)

model.n_estimators *= 2
model_reinforced = model.fit(X_train, y_train)




