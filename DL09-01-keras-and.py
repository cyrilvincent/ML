import numpy as np
import tensorflow as tf
tf.random.set_seed(1)

X=np.array([[0,0],[0,1],[1,0],[1,1]])
y=np.array([0,0,0,1])

model = tf.keras.Sequential([
    tf.keras.layers.Dense(4, input_shape=(X.shape[1],)),
    tf.keras.layers.Dense(8),
    tf.keras.layers.Dense(1)
  ])

model.compile(loss="mse")
model.summary()

history = model.fit(X, y, epochs=100, batch_size=1)

res = model.predict(X)
print(res)
print(res > 0.5)

