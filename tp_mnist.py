import numpy as np
import sklearn.neighbors as n
import matplotlib.pyplot as plt
import sklearn.ensemble as rf
import pickle

np.random.seed(42)

with np.load("data/mnist/mnist.npz", allow_pickle=True) as f:
    xtrain, ytrain = f["x_train"], f["y_train"]
    xtest, ytest = f["x_test"], f["y_test"]

print(xtest.shape, ytrain.shape)

xtrain = xtrain.reshape(-1, 28*28)
xtest = xtest.reshape(-1, 28*28)

# model = n.KNeighborsClassifier(n_neighbors=3)
model = rf.RandomForestClassifier()
model.fit(xtrain, ytrain)
print(model.score(xtest, ytest))
ypredicted = model.predict(xtest)

with open("data/mnist/rf.pickle", "wb") as f:
    pickle.dump(model, f)

plt.imshow(model.feature_importances_.reshape(28,28))
plt.show()



xtest = xtest.reshape(-1, 28, 28)
select = np.random.randint(xtest.shape[0], size=12)



for index, value in enumerate(select):
    plt.subplot(3,4,index+1)
    plt.axis("off")
    plt.imshow(xtest[value], cmap=plt.cm.gray_r)
    plt.title(f'Predicted {ypredicted[value]}')
plt.show()

errors = ytest != ypredicted
xerrors = xtest[errors]
yerrors = ypredicted[errors]

select = np.random.randint(xerrors.shape[0], size=12)

for index, value in enumerate(select):
    plt.subplot(3,4,index+1)
    plt.axis("off")
    plt.imshow(xerrors[value], cmap=plt.cm.gray_r)
    plt.title(f'Predicted {yerrors[value]}')
plt.show()
