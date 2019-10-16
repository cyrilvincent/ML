from sklearn.datasets import load_breast_cancer
import tensorflow.compat.v1 as tf
cancer = load_breast_cancer() # more info : https://goo.gl/U2Uwz2
X=cancer['data']
y=cancer['target']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
print(X_train.shape)
print(X_test)

#from sklearn.neural_network import MLPClassifier
#mlp = MLPClassifier(hidden_layer_sizes=(30,30,30))
#import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.compat.v1 as tf
import numpy as np
import keras.backend as K


def binary_recall_specificity(y_true, y_pred, recall_weight, spec_weight):
    # Pénalise les faux negatifs
    # Recall

    TN = np.logical_and(K.eval(y_true) == 0, K.eval(y_pred) == 0)
    TP = np.logical_and(K.eval(y_true) == 1, K.eval(y_pred) == 1)

    FP = np.logical_and(K.eval(y_true) == 0, K.eval(y_pred) == 1)
    FN = np.logical_and(K.eval(y_true) == 1, K.eval(y_pred) == 0)

    # Converted as Keras Tensors
    TN = K.sum(K.variable(TN))
    FP = K.sum(K.variable(FP))

    specificity = TN / (TN + FP + K.epsilon())
    recall = TP / (TP + FN + K.epsilon())

    return 1.0 - (recall_weight*recall + spec_weight*specificity)

def custom_loss(recall_weight, spec_weight):

    def recall_spec_loss(y_true, y_pred):
        return binary_recall_specificity(y_true, y_pred, recall_weight, spec_weight)

    # Returns the (y_true, y_pred) loss function
    return recall_spec_loss

loss = custom_loss(recall_weight=0.9, spec_weight=0.1)

model = keras.Sequential([
    keras.layers.Dense(30, activation=tf.nn.relu,
                       input_shape=(X_train.shape[1],)),
    keras.layers.Dense(30, activation=tf.nn.relu),
    keras.layers.Dense(30, activation=tf.nn.relu),
    keras.layers.Dense(30, activation=tf.nn.relu),
    keras.layers.Dense(1)
  ])

# model = keras.Sequential()
# model.add(keras.layers.Dense(30, activation=tf.nn.relu,
#                        input_shape=(X_train.shape[1],)))
# model.add(keras.layers.Dense(30, activation=tf.nn.relu))
# model.add(keras.layers.Dense(30, activation=tf.nn.relu))
# model.add(keras.layers.Dense(30, activation=tf.nn.relu))
# model.add(keras.layers.Dense(1))

#model.compile(loss="mse", optimizer="sgd")
#sgd = keras.optimizers.SGD(nesterov=True, lr=1e-5)
model.compile(loss=loss, optimizer="adam")
model.summary()

history = model.fit(X_train, y_train, epochs=2000)
eval = model.evaluate(X_test, y_test)
print(eval)
model.save("cancer.h5")
