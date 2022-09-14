import tensorflow as tf
import tensorflow.keras as tfk

from tensorflow.keras.layers import Dense

def create_model():
    m = tfk.Sequential(
        [
            Dense(3, activation='relu', input_shape=(1,)),
            Dense(1)
        ]
    )
    m.compile(loss='mse', metrics=['mae'])
    return m