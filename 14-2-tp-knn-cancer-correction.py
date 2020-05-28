from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer() # more info : https://goo.gl/U2Uwz2

#input
x=cancer['data']
y=cancer['target']

print(x.shape) #569 * 30
print(y.shape) #569

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y)

import sklearn.neighbors as nn
bestk = 0
bestscore = 0
for k in range(3, 8):
    model = nn.KNeighborsClassifier(k)
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    print(f"k={k} score={score}")
    if score > bestscore:
        bestk = k
        bestscore = score

print(f"Best k={bestk} score={bestscore}")
model = nn.KNeighborsClassifier(bestk)
model.fit(x_train, y_train)
predicted = model.predict(x_test)
print(predicted)
print(y_test)
print(predicted - y_test)

import numpy as np
softmax = lambda x : np.exp(x) / sum(np.exp(x))
geomax = lambda x : (x ** 2) / sum(x ** 2)
linearmax = lambda x : x / sum(x)

l = np.array([0.5,0.9,0.95])
print(linearmax(l))
print(geomax(l))
print(softmax(l))
