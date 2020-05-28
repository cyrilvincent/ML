from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer() # more info : https://goo.gl/U2Uwz2

#input
x=cancer['data']
y=cancer['target']

print(x.shape) #569 * 30
print(y.shape) #569

import sklearn.model_selection as ms
x_train,x_test,y_train,y_test = ms.train_test_split(x,y)
