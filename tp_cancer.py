import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn.preprocessing as pp

# Charger data/breast-cancer/data.csv dans un dataset
# Faire un describe sur la colonne perimeter_worst
# Sauvegarder le dataset en HTML

dataframe = pd.read_csv("data/breast-cancer/data.csv")
print(dataframe["perimeter_worst"].describe())
dataframe.to_html("data/breast-cancer/data.html")
# plt.matshow(dataframe.corr())
# plt.show()



y = dataframe.diagnosis
x = dataframe.drop(["diagnosis", "id"], axis=1)

# train_test_split

scaler = pp.StandardScaler()
scaler.fit(x)
xnorm = scaler.transform(x)
print(xnorm)

# TODO
# train_test_split
# normalisé x
# knn
# predict
# score





# print(x["perimeter_mean"])
# mean = np.mean(x["perimeter_mean"])
# std = np.std(x["perimeter_mean"])
# print(f"Mean: {mean}, std:{std}")
# norm = (x["perimeter_mean"] - mean) / std
# print(norm)
#
# print(x["smoothness_se"])
# mean = np.mean(x["smoothness_se"])
# std = np.std(x["smoothness_se"])
# print(f"Mean: {mean}, std:{std}")
# norm = (x["smoothness_se"] - mean) / std
# print(norm)



