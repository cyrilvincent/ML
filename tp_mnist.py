import sklearn
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import sklearn.preprocessing as pp
import sklearn.model_selection as ms
import numpy as np
import sklearn.neighbors as n
import sklearn.ensemble as rf
import sklearn.tree as tree
import pickle

data = np.load("data/mnist/mnist.npz")
print(data)
xtrain = data["x_train"]
xtest = data["x_test"]
ytrain = data["y_train"]
ytest = data["y_test"]



print(xtrain.shape, xtest.shape)


np.random.seed(42)

xtrain = xtrain.reshape(-1, 28*28)

for pixel in xtrain[0]:
    print(f"{pixel},", end="")

xtest = xtest.reshape(-1, 28*28)
print(xtest.shape)

# model = n.KNeighborsClassifier(n_neighbors=3)
model = rf.RandomForestClassifier(max_depth=6)
model.fit(xtrain, ytrain)



ypred = model.predict(xtest)

train_score = model.score(xtrain, ytrain)
test_score = model.score(xtest, ytest)

with open(f"data/mnist/rf-{test_score:.2f}.pkl", "wb") as f:
    pickle.dump(model, f)

print(train_score, test_score)

tree.export_graphviz(model.estimators_[0], out_file="data/mnist/tree.dot", feature_names=[str(x) for x in range(784)], class_names=[str(x) for x in range(10)])

plt.imshow(model.feature_importances_.reshape(28,28))
plt.show()

xtest = xtest.reshape(-1, 28, 28)
select = np.random.randint(xtest.shape[0], size=12)

for index, value in enumerate(select):
    plt.subplot(3,4, index + 1)
    plt.axis("off")
    plt.imshow(xtest[value], cmap=plt.cm.gray_r)
    plt.title(f"Predicted {ypred[value]}")
plt.show()

errors = ytest != ypred
xerrors = xtest[errors]
yerrors = ypred[errors]

select = np.random.randint(xerrors.shape[0], size=12)

for index, value in enumerate(select):
    plt.subplot(3,4, index + 1)
    plt.axis("off")
    plt.imshow(xerrors[value], cmap=plt.cm.gray_r)
    plt.title(f"Predicted {yerrors[value]}")
plt.show()

# refaire tourner tp_cencer.py et analyser l'arbre et les features importances
# Porter tp_mnist avec RandomForest
# Afficher les Features importances features_importances_.reshape(28, 28) plt.imshow()

