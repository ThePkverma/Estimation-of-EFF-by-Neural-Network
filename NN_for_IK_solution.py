import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import relu,linear
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
f = np.load('data.npy')
f = f[1:, :]

x = f[:, 0:3]

y = f[:, 3:6]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
tf.random.set_seed(1234)

model = Sequential(
    [  
        ### START CODE HERE ### 
        tf.keras.Input(shape=(3,)),
        Dense(120, activation='relu',name='L1'),
        Dense(40, activation='relu',name='L2'),
        Dense(3, activation='linear',name='L3')
        ### END CODE HERE ### 

    ], name="Complex"
)
model.compile(
    ### START CODE HERE ### 
    loss='mse',
    optimizer=tf.keras.optimizers.Adam(0.1),
    ### END CODE HERE ### 
)
model.summary()
model.predict(X_test)
f=model.weights
np.array([f]).shape
X_train.shape
model.fit(
    X_train,
    y_train,
    batch_size=64,
    epochs=20,
    validation_data=(X_test, y_test),
)
model.summary()
model.predict(X_test)
f=model.weights
np.array([f]).shape
X_train.shape
model.fit(
    X_train,
    y_train,
    batch_size=64,
    epochs=20,
    validation_data=(X_test, y_test),
)

f=model.weights
np.array([f]).shape
X_train.shape