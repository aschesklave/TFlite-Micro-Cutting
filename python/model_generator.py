import os

import tensorflow as tf
import tensorflow.keras as tfk

from tensorflow.keras.layers import Conv2D, Dense, Flatten

NUM_CLASSES = 10

def convert_model(model: tfk.Sequential):
    path = 'model'
    model.save(path)
    generic_converter = tf.lite.TFLiteConverter.from_saved_model(path)
    generic_converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    generic_converter.experimental_enable_resource_variables = True
    generic_tflite_m = generic_converter.convert()
    with open(f'{path}.tflite', 'wb') as f:
        f.write(generic_tflite_m)


def train_simple_model():
    val_split = 0.2
    batch_size = 100
    epochs = 15
    optimizer = 'adam'
    loss = 'categorical_crossentropy'
    metrics = 'accuracy'

    img_shape = x_train[0].shape
    model = tfk.Sequential([
        tfk.Input(shape=img_shape),
        Conv2D(12, kernel_size=3, activation='relu'),
        Flatten(),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.fit(x_train_norm, y_train, batch_size=batch_size, epochs=epochs, validation_split=val_split)
    evaluation = model.evaluate(x_test_norm, y_test_categorical, batch_size=batch_size)
    return model


def generate_image_cpp_files(x, y, num_images=10):
    file_name = 'images'
    file_path = 'C:\\Users\\Julian\\Documents\\PlatformIO\\Projects\\alto4arduino\\src\\'
    img_size = 28
    cpp_file = f'#include "{file_name}.h"\n\nconst unsigned int size = {img_size};\n\n'
    header_file = f'#pragma once\n\nextern const unsigned int size;\n\n'

    for img_no in range(num_images):
        header_file += f'extern const unsigned int y_{img_no};\n'
        header_file += f'extern const unsigned char img_{img_no}[{img_size * img_size}];\n\n'

        cpp_file += f'const unsigned int y_{img_no} = {y[img_no]};\n'
        cpp_file += f'const unsigned char img_{img_no}[{img_size * img_size}] = {{\n'
        for i in range(img_size):
            for j in range(img_size):
                cpp_file += f'0x{int(x[img_no][i][j]):02x}, '
            cpp_file += '\n'

        cpp_file = cpp_file[:-3] + '};\n\n'

    cpp_f = open(f'{file_path}{file_name}.cpp', 'w')
    header_f = open(f'{file_path}{file_name}.h', 'w')
    cpp_f.write(cpp_file)
    header_f.write(header_file)


(x_train, y_train), (x_test, y_test) = tfk.datasets.fashion_mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_train_norm = x_train / 255.0
x_test_norm = x_test / 255.0
y_train = tfk.utils.to_categorical(y_train, NUM_CLASSES)
y_test_categorical = tfk.utils.to_categorical(y_test, NUM_CLASSES)

simple_model = train_simple_model()
convert_model(simple_model)
#generate_image_cpp_files(x_test, y_test)