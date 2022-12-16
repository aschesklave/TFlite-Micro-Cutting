import keras.models
import tensorflow as tf
import tensorflow.keras as tfk
import numpy as np
import math
import tensorflow_datasets as tfds

from model_generator import convert_model, convert_model_int

from tensorflow.keras.layers import Dense, Input

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


def create_model(load, model_path):
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

def build_simple_model():
    model_path = 'reds'
    load = False
    simple_model = create_model(load, model_path)
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


def flatten_image(x):
    padded = np.pad(x, ((2,2), (2,2)))
    return padded.flatten().reshape(32*32).astype('float32') / 255


def flatten_image_reds(x, y):
    x = x / 255
    x = (x - 0.1307) / 0.3081  # subtract mean and divide by standard deviation
    padded = tf.pad(tensor=x, paddings=tf.constant([[0, 0], [2, 2], [2, 2], [0, 0]]))
    return tf.cast(x=tf.reshape(tensor=padded, shape=[-1, 1024]), dtype=tf.float32), y


def get_datapoints(ds):
    data = tfds.as_numpy(ds.take(1))
    for d in data:
        return d[0], d[1]


def generate_reds_image_cpp_files(num_images=10):
    train_data, val_data, test_data = tfds.load("mnist",
                                                split=['train[10000:]', 'train[0:10000]', 'test'],
                                                batch_size=num_images, as_supervised=True)

    val_data = val_data.map(flatten_image_reds)

    file_name = 'images'
    file_path = 'C:\\Users\\Julian\\Documents\\PlatformIO\\Projects\\alto4arduino\\'
    img_size = 32
    cpp_file = f'#include "{file_name}.h"\n\nconst unsigned int size = {img_size};\n\n'
    header_file = f'#pragma once\n\nextern const unsigned int size;\n\n'
    x, y = get_datapoints(val_data)

    for img_no in range(num_images):
        header_file += f'extern const unsigned int y_{img_no};\n'
        header_file += f'extern const float img_{img_no}[{img_size * img_size}];\n\n'

        cpp_file += f'const unsigned int y_{img_no} = {y[img_no]};\n'
        cpp_file += f'const float img_{img_no}[{img_size * img_size}] = {{\n'
        for i in range(img_size * img_size):
            cpp_file += f'{x[img_no][i]}, '.rjust(25)
            if (i+1) % 5 == 0:
                cpp_file += '\n'

        cpp_file = cpp_file[:-3] + '};\n\n'

    cpp_f = open(f'{file_path}src\\{file_name}.cpp', 'w')
    header_f = open(f'{file_path}include\\{file_name}.h', 'w')
    cpp_f.write(cpp_file)
    header_f.write(header_file)


def build_reds_model():
    image_size = 32*32
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = np.array(list(map(flatten_image, x_train)))
    x_test = np.array(list(map(flatten_image, x_test)))
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    m = tfk.Sequential(
        [
            Dense(19, activation='relu', input_shape=(image_size,)),
            Dense(10, activation="softmax"),
        ]
    )
    m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    m.fit(x_train, y_train, epochs=30, batch_size=128)
    evaluation = m.evaluate(x_test, y_test, batch_size=128)
    convert_model(m, 'custom_reds')


#build_reds_model()
generate_reds_image_cpp_files()