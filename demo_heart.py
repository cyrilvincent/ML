import pandas as pd

pd.options.display.width = 0
dataframe = pd.read_csv("data/heartdisease/data_cleaned_up.csv")

ok = dataframe[dataframe.num == 0]
ko = dataframe[dataframe.num == 1]
print(ok.chol.describe())
print(ko.chol.describe())

print(dataframe.corr())