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
        Conv2D(24, kernel_size=3, activation='relu'),
        Flatten(),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.fit(x_train_norm, y_train, batch_size=batch_size, epochs=epochs, validation_split=val_split)
    evaluation = model.evaluate(x_test_norm, y_test_categorical, batch_size=batch_size)
    return model


(x_train, y_train), (x_test, y_test) = tfk.datasets.fashion_mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_train_norm = x_train / 255.0
x_test_norm = x_test / 255.0
y_train = tfk.utils.to_categorical(y_train, NUM_CLASSES)
y_test_categorical = tfk.utils.to_categorical(y_test, NUM_CLASSES)

simple_model = train_simple_model()
convert_model(simple_model)