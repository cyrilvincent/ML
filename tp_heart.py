import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataframe = pd.read_csv("data/heartdisease/data_cleaned_up.csv")
# y = dataframe["num"]
# x = dataframe.drop("num", 1)

plt.matshow(dataframe.corr())
plt.show()

ok = dataframe[dataframe["num"] == 0]
ko = dataframe[dataframe["num"] == 1]

print(f"OK chol mean: {np.mean(ok.chol)}, std: {np.std(ok.chol)}")
print(f"KO chol mean: {np.mean(ko.chol)}, std: {np.std(ko.chol)}")

print(f"OK sex mean: {np.mean(ok.sex)}, std: {np.std(ok.sex)}")
print(f"KO sex mean: {np.mean(ko.sex)}, std: {np.std(ko.sex)}")

print(f"OK age mean: {np.mean(ok.age)}, std: {np.std(ok.age)}")
print(f"KO age mean: {np.mean(ko.age)}, std: {np.std(ko.age)}")
print(f"Dataframe age mean: {np.mean(dataframe.age)}, std: {np.std(dataframe.age)}")

# afficher les correlations sur le dataframe
# ok = filtrer sur num == 0
# ko = filtrer sur num == 1
# sur chol calculer mean, std pour ok et ko => conclusion
# idem sur sex et age

