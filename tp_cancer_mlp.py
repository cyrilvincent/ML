import pandas as pd
import numpy as np
import sklearn.linear_model as lm
import sklearn.neighbors as nn
import sklearn.model_selection as ms
import sklearn.ensemble as rf
import matplotlib.pyplot as plt
import sklearn.tree as tree
import pickle
import sklearn.neural_network as neural

np.random.seed(0)
dataframe = pd.read_csv("data/breast-cancer/data.csv", index_col="id")

# Afficher le describe
# Afficher diagnosis, radius_mean, concavity_se
# Filtrer les patients sains et malades
# Afficher la moyenne (np.mean()) de radius_mean, concavity_se pour les 2 groupes de patients
sains = dataframe[dataframe.diagnosis == 0]
malades = dataframe[dataframe.diagnosis == 1]
print(np.mean(sains.radius_mean), np.mean(malades.radius_mean))
print(np.mean(sains.concavity_se), np.mean(malades.concavity_se))

y = dataframe.diagnosis
x = dataframe.drop("diagnosis", axis=1)
print(x)

xtrain, xtest, ytrain, ytest = ms.train_test_split(x,y,train_size=0.8,test_size=0.2)

# # model = lm.LinearRegression()
# for k in range(3, 11, 2):
#     model = nn.KNeighborsClassifier(n_neighbors=k)
#     model.fit(xtrain, ytrain)
#     print(k, model.score(xtrain, ytrain))
#     print(k, model.score(xtest, ytest))

# model = rf.RandomForestClassifier(warm_start=True)
model = neural.MLPClassifier((30, 20, 10), max_iter=10000, activation="relu")
model.fit(xtrain, ytrain)

print(model.score(xtest, ytest))

with open("data/breast-cancer/model-mlp.pickle", "wb") as f:
    pickle.dump(model, f)

model = None

with open("data/breast-cancer/model-mlp.pickle", "rb") as f:
    model = pickle.load(f)
