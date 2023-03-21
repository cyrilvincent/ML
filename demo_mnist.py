import pickle

import numpy as np
import sklearn.neural_network as neural
import sklearn.model_selection as ms
import sklearn.metrics as metrics
import sklearn.neighbors as ne
import sklearn.ensemble as rf
from sklearn.tree import export_graphviz


np.random.seed(0)

with np.load("data/mnist/mnist.npz", allow_pickle=True) as f:
    x_train, y_train = f["x_train"], f["y_train"] # 60000
    x_test, y_test = f["x_test"], f["y_test"] # 10000
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

x_train = x_train.reshape(-1, 28*28)
x_test = x_test.reshape(-1, 28*28)

# x_train, x_test, y_train, y_test = ms.train_test_split(x_train, y_train, train_size=0.1, test_size=0.9)

model = rf.RandomForestClassifier(warm_start=True)
model.fit(x_train, y_train)

with open("data/mnist/rf.pickle", "wb") as f:
    pickle.dump(model, f)

model = None

with open("data/mnist/rf.pickle", "rb") as f:
    model = pickle.load(f)

print(model.score(x_test, y_test))
y_pred = model.predict(x_test)

print(metrics.classification_report(y_test, y_pred))
print(metrics.confusion_matrix(y_test, y_pred))



export_graphviz(model.estimators_[0],
                out_file='data/mnist/tree.dot',
                # feature_names = x_train.feature_names,
                # class_names = y_train.target_names,
                rounded = True, proportion = False,
                precision = 2, filled = True)

predicted = model.predict(x_test)

images = x_test.reshape((-1, 28, 28))

# On selectionne un echantillon de 12 images au hasard
select = np.random.randint(images.shape[0], size=12)

# On affiche les images avec la prédiction associée
import matplotlib.pyplot as plt
for index, value in enumerate(select):
    plt.subplot(3,4,index+1)
    plt.axis('off')
    plt.imshow(images[value],cmap=plt.cm.gray_r,interpolation="nearest")
    plt.title('Predicted: %i' % predicted[value])

plt.show()

# Gestion des erreurs
# on récupère les données mal prédites
misclass = (y_test != predicted)
misclass_images = images[misclass,:,:]
misclass_predicted = predicted[misclass]

# on sélectionne un échantillon de ces images
select = np.random.randint(misclass_images.shape[0], size=12)

# on affiche les images et les prédictions (erronées) associées à ces images
for index, value in enumerate(select):
    plt.subplot(3,4,index+1)
    plt.axis('off')
    plt.imshow(misclass_images[value],cmap=plt.cm.gray_r,interpolation="nearest")
    plt.title('Predicted: %i' % misclass_predicted[value])

plt.show()

