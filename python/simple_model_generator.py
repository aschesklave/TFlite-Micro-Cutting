import keras.models
import tensorflow as tf
import tensorflow.keras as tfk
import numpy as np
import math

from model_generator import convert_model, convert_model_int

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
    if load:
        return keras.models.load_model(model_path)
    else:
        m = tfk.Sequential(
            [
                # Dense(16, activation='relu', input_shape=(2,)),
                Dense(16, activation='relu', input_shape=(1,)),
                Dense(16, activation='relu'),
                Dense(1)
            ]
        )
        m.compile(loss='mse', metrics=['mae'])
        return m


model_path = 'custom_model_float'
load = True
simple_model = create_model()
if load:
    x = np.array((0.5,))
    target_layer = simple_model.layers[1]
    weights = target_layer.get_weights()
    inputs = keras.Model(inputs=simple_model.input, outputs=simple_model.layers[0].output).predict(x)

    weights[0][:, 8:] = 0
    weights[1][8:] = 0
    simple_model.layers[1].set_weights(weights)
    res = simple_model.predict(x)
    inter_res = keras.Model(inputs=simple_model.input, outputs=simple_model.layers[1].output).predict(x)
    print(res[0])
else:
    x_train, x_test, x_validate, y_train, y_test, y_validate = generate_dataset()
    simple_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # For shape (2,)
    # x_train = np.array([x_train, x_train]).T
    # y_train = np.array([y_train, y_train]).T
    # x_validate = np.array([x_validate, x_validate]).T
    # y_validate = np.array([y_validate, y_validate]).T

    simple_model.fit(x_train, y_train, epochs=500, batch_size=64, validation_data=(x_validate, y_validate))
    #convert_model_int(simple_model, 'custom_model', x_train)
    convert_model(simple_model, model_path)