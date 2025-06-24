import pandas as pd
import sklearn.linear_model as lm
import sklearn.neighbors as nn
import numpy as np
import sklearn.ensemble as rf
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
import pickle

pd.set_option('display.max_columns', None)
dataframe = pd.read_csv("data/heartdisease/data_cleaned_up.csv")
dataframe["rnd"] = np.random.rand(260)
print(dataframe.describe())

y = dataframe.num
x = dataframe.drop("num", axis=1)

# np.random.seed(42)
# # model = lm.LinearRegression()
# for i in range(3,12):
#     model = nn.KNeighborsClassifier(n_neighbors=i)
#     model.fit(x, y)
#     print(model.score(x, y))

model = rf.RandomForestClassifier()
model.fit(x, y)
score = model.score(x, y)
print(score)

with open(f"data/heartdisease/rf-{int(score*100)}.pickle", "wb") as f:
    pickle.dump(model, f)



export_graphviz(model.estimators_[0], out_file="data/heartdisease/tree.dot", feature_names=x.columns, class_names=["0", "1"])

print(model.feature_importances_)

plt.bar(x.columns, model.feature_importances_)
plt.xticks(rotation=45)
plt.show()



