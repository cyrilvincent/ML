import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import sklearn.preprocessing as pp
import sklearn.pipeline as pipe
import sklearn.model_selection as ms
import numpy as np
import sklearn.neighbors as n
import sklearn.ensemble as rf
import sklearn.tree as tree
import pickle

data = np.load("data/mnist/mnist.npz")
xtrain = data["x_train"] #[::10]
xtest = data["x_test"]
ytrain = data["y_train"]
ytest = data["y_test"]

print(xtrain.shape)

xtrain = xtrain.reshape(-1, 28*28)
xtest = xtest.reshape(-1, 28*28)

# model = n.KNeighborsClassifier(n_neighbors=3)
model = rf.RandomForestClassifier()
model.fit(xtrain, ytrain)

print(f"Train score: {model.score(xtrain, ytrain)}")
print(f"Test score: {model.score(xtest, ytest)}")

with open(f"data/mnist/rf-{model.score(xtest, ytest):.3f}.pkl", "wb") as f:
    pickle.dump(model, f)

ypred = model.predict(xtest)

xtest = xtest.reshape(-1, 28, 28)
select = np.random.randint(xtest.shape[0], size=12)

# for index, value in enumerate(select):
#     plt.subplot(3, 4, index + 1)
#     plt.axis("off")
#     plt.imshow(xtest[value], cmap=plt.cm.gray_r)
#     plt.title(f"Predicted {ypred[value]}")
# plt.show()

plt.imshow(model.feature_importances_.reshape(28, 28))
plt.show()

# RF
# Save
# Faire un predict dans un autre fichier et afficher les 12 premiers xtest[:12]
# Afficher les features_importances dans une heat map
# feature.reshape(28,28) + imshow

# with open(f"data/breast-cancer/rf-{model.score(xtest, ytest):.2f}.pkl", "wb") as f:
#     pickle.dump([scaler, model], f)


