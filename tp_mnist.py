import numpy as np
import sklearn.neighbors as nn
import matplotlib.pyplot as plt
import sklearn.ensemble as rf
from sklearn.tree import export_graphviz
import sklearn.neural_network as nn

np.random.seed(42)

with np.load("data/mnist/mnist.npz", allow_pickle=True) as f:
    xtrain, ytrain = f["x_train"], f["y_train"]
    xtest, ytest = f["x_test"], f["y_test"]

print(xtrain.shape, ytrain.shape, xtest.shape, ytest.shape)
xtrain = xtrain.reshape(-1, 28*28) # 784
xtest = xtest.reshape(-1, 28*28)

# model = rf.RandomForestClassifier()
model = nn.MLPClassifier(hidden_layer_sizes=(500,200,100))
# model = rf.GradientBoostingClassifier()
model.fit(xtrain, ytrain)
ypredicted = model.predict(xtest)
score = model.score(xtest, ytest)
print(f"Score: {score:.3f}")
#
# export_graphviz(model.estimators_[0], out_file="data/mnist/tree.dot", class_names=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
#
# matrix = model.feature_importances_.reshape(28, 28)
# plt.matshow(matrix)
# plt.show()
#
# xtest = xtest.reshape(-1, 28, 28)
# select = np.random.randint(xtest.shape[0], size=12)
#
# for index, value in enumerate(select):
#     plt.subplot(3, 4, index + 1)
#     plt.axis("off")
#     plt.imshow(xtest[value], cmap=plt.cm.gray_r, interpolation="nearest")
#     plt.title(f"Predicted {ypredicted[value]}")
# plt.show()
#
# errors = ytest != ypredicted
# xerrors = xtest[errors]
# yerrors = ypredicted[errors]
#
# select = np.random.randint(xerrors.shape[0], size=12)
#
# for index, value in enumerate(select):
#     plt.subplot(3, 4, index + 1)
#     plt.axis("off")
#     plt.imshow(xerrors[value], cmap=plt.cm.gray_r, interpolation="nearest")
#     plt.title(f"Predicted {yerrors[value]}")
# plt.show()


