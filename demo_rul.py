import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.model_selection as ms
import sklearn.preprocessing as pp
import sklearn.neural_network as nn
import sklearn.neighbors as neighbors
import sklearn.ensemble as rf
import sklearn.svm as svm
import tensorflow as tf

pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 200)
dataframe = pd.read_csv("data/rul/Battery_RUL.csv")
print(dataframe.head())
print(dataframe.describe())
print(dataframe.shape)

# plt.title('RUL, Remaining Useful Time Histogram')
# sns.histplot(dataframe.RUL, kde=True)
# plt.show()

# correlations = dataframe.corr()
# plt.figure(figsize = (15,8))
# sns.heatmap(correlations, annot=True, cbar=False, cmap='Blues', fmt='.1f')
# plt.show()
#
# plt.figure(figsize=(10, 6))
# plt.scatter(dataframe['Max. Voltage Dischar. (V)'], dataframe['RUL'], alpha=0.5)
# plt.title('Scatter Plot: Max. Voltage Discharge vs. Remaining Useful Lifetime')
# plt.xlabel('Max. Voltage Dischar. (V)')
# plt.ylabel('Remaining Useful Lifetime (RUL)')
# plt.grid(True)
# plt.show()
#
# plt.figure(figsize=(10, 6))
# plt.scatter(dataframe['Min. Voltage Charg. (V)'], dataframe['RUL'], alpha=0.5)
# plt.title('Scatter Plot: Min. Voltage Charge vs. Remaining Useful Lifetime')
# plt.xlabel('Min. Voltage Charg. (V)')
# plt.ylabel('Remaining Useful Lifetime (RUL)')
# plt.grid(True)
# plt.show()
#
# plt.figure(figsize=(10, 6))
# plt.scatter(dataframe['Time at 4.15V (s)'], dataframe['RUL'], alpha=0.5)
# plt.title('Scatter Plot: Time at 4.15V vs. Remaining Useful Lifetime')
# plt.xlabel('Time at 4.15V (s)')
# plt.ylabel('Remaining Useful Lifetime (RUL)')
# plt.grid(True)
# plt.show()

y = dataframe['RUL']
x = dataframe.drop(['RUL'], axis=1)
print(x.head())

np.random.seed(42)
x_train, x_test, y_train, y_test = ms.train_test_split(x, y)

y_train = y_train.values.reshape(-1, 1)
y_test = y_test.values.reshape(-1, 1)
scaler = pp.RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
yscaler = pp.MinMaxScaler()
yscaler.fit(y_train)
y_train = yscaler.transform(y_train)
y_test = yscaler.transform(y_test)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation="relu", input_shape=(x_train.shape[1],)),
    tf.keras.layers.Dense(10, activation="relu"),
    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid),
  ])
model.compile(loss="mse")
trained = model.fit(x_train, y_train, epochs=10, batch_size=1,validation_data=(x_test, y_test))
print(model.summary())
score = model.evaluate(x_test, y_test)
print(score, score ** 0.5, score ** 0.5 * 500)

ypred = model.predict(x_test)
ypred = yscaler.inverse_transform(ypred)
print(ypred)
