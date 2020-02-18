import numpy as np

with np.load("data/mnist.npz", allow_pickle=True) as f:
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']

sample = np.random.randint(60000, size=5000)
data = x_train[sample]
target = y_train[sample]

from sklearn.neural_network import MLPClassifier
mlp = None #TODO
mlp.fit(x_train,y_train)
print(mlp.score(x_test, y_test))
# Le score n'est pas terrible, nous verrons pourquoi plus tard (MSE ??)


