import tensorflow as tf
import tensorflow.keras as tfk
import numpy as np
import math

from model_generator import convert_model

from tensorflow.keras.layers import Dense

def generate_dataset():
    SAMPLES = 1000
    x_values = np.random.uniform(
        low=0, high=2 * math.pi, size=SAMPLES).astype(np.float32)
    np.random.shuffle(x_values)
    y_values = np.sin(x_values).astype(np.float32)
    y_values += 0.1 * np.random.randn(*y_values.shape)
    TRAIN_SPLIT = int(0.6 * SAMPLES)
    TEST_SPLIT = int(0.2 * SAMPLES + TRAIN_SPLIT)
    x_tr, x_te, x_val = np.split(x_values, [TRAIN_SPLIT, TEST_SPLIT])
    y_tr, y_te, y_val = np.split(y_values, [TRAIN_SPLIT, TEST_SPLIT])
    return x_tr, x_te, x_val, y_tr, y_te, y_val


def create_model():
    m = tfk.Sequential(
        [
            Dense(8, activation=None, input_shape=(1,), use_bias=False),
            Dense(1)
        ]
    )
    m.compile(loss='mse', metrics=['mae'])
    return m


simple_model = create_model()
x_train, x_test, x_validate, y_train, y_test, y_validate = generate_dataset()
simple_model.fit(x_train, y_train, epochs=500, batch_size=64, validation_data=(x_validate, y_validate))
convert_model(simple_model, 'simple_model_8')