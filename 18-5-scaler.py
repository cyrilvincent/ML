import sklearn.preprocessing as pp
data = [[0, 0], [0, 0], [1, 1], [1, 1]]
scaler = pp.StandardScaler()
scaler.fit(data)
print(scaler.mean_) # [0.5 0.5]
import math
print([math.sqrt(x) for x in scaler.var_]) # [0.5 0.5]
print(scaler.transform(data))

standardSclaler = lambda x, mean, std : (x - mean) / std
print(standardSclaler(2,1,0.5))

