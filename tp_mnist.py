import numpy as np
import sklearn.neighbors as nn
import matplotlib.pyplot as plt
import sklearn.ensemble as rf

np.random.seed(42)

with np.load("data/mnist/mnist.npz", allow_pickle=True) as f:
    xtrain, ytrain = f["x_train"], f["y_train"]
    xtest, ytest = f["x_test"], f["y_test"]

print(xtrain.shape, ytrain.shape, xtest.shape, ytest.shape)
xtrain = xtrain.reshape(-1, 28*28)
xtest = xtest.reshape(-1, 28*28)

# for k in range(3, 10):
#     model = nn.KNeighborsClassifier(n_neighbors=k)
#     model.fit(xtrain, ytrain)
#     score = model.score(xtrain, ytrain)
#     print(f"Score {k} train: {score:.3f}")
#     score = model.score(xtest, ytest)
#     print(f"Score {k} test: {score:.3f}")
#     ypredicted = model.predict(xtest)

model = rf.RandomForestClassifier()
model.fit(xtrain, ytrain)
score_train = model.score(xtrain, ytrain)
score_test = model.score(xtest, ytest)
print(f"Train score: {score_train:.2f}")
print(f"Test score: {score_test:.2f}")
ypredicted = model.predict(xtest)

xtest = xtest.reshape(-1, 28, 28)
select = np.random.randint(xtest.shape[0], size=12)

matrix = model.feature_importances_.reshape(28, 28)
plt.matshow(matrix)
plt.show()

for index, value in enumerate(select):
    plt.subplot(3, 4, index + 1)
    plt.axis("off")
    plt.imshow(xtest[value], cmap=plt.cm.gray_r, interpolation="nearest")
    plt.title(f"Predicted {ypredicted[value]}")
plt.show()

errors = ytest != ypredicted
xerrors = xtest[errors]
yerrors = ypredicted[errors]

select = np.random.randint(xerrors.shape[0], size=12)

for index, value in enumerate(select):
    plt.subplot(3, 4, index + 1)
    plt.axis("off")
    plt.imshow(xerrors[value], cmap=plt.cm.gray_r, interpolation="nearest")
    plt.title(f"Predicted {yerrors[value]}")
plt.show()


