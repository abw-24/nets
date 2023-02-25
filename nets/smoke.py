
"""
Module for simple smoke tests with toy MNIST data and hardcoded configurations.
"""
#TODO: Replace with more mature unit and integration testing


import tensorflow as tf

from nets.utils import get_obj
from nets.models.factory import MLPFactory, VAEFactory


def mlp():
    """
    Test mlp on mnist data.
    """

    # load mnist data, flatten, and normalize to 0-1
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape((x_train.shape[0], 784))
    x_train = x_train / 255.0

    # create a batch feed from the train tensors
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)) \
        .shuffle(10000) \
        .batch(32)

    config = {
        "hidden_dims": [32, 16],
        "activation": "relu",
        "output_dim": 10,
        "output_activation": "softmax",
        "optimizer": {"Adam": {"learning_rate": 0.001}},
        "loss": {"SparseCategoricalCrossentropy": {}},
        "epochs": 2
    }

    model = MLPFactory.apply(config)
    model.build(input_shape=(None, 784))
    model.compile(
            loss=get_obj(tf.keras.losses, config["loss"]),
            optimizer=get_obj(tf.keras.optimizers, config["optimizer"])
    )

    for _ in range(config["epochs"]):
        for x, y in train_ds:
            model.train_on_batch(x, y)

        print("{model} epoch metrics: {metrics}".format(
                model=model.name,
                metrics=model.get_metrics_result()
        ))


def vae():
    """
    Test vae on mnist data.
    """

    # load mnist data, flatten, and normalize to 0-1
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape((x_train.shape[0], 784))
    x_train = x_train / 255.0

    # create a batch feed from the train tensors
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)) \
        .shuffle(10000) \
        .batch(32)

    config = {
        "encoding_dims": [8],
        "latent_dim": 4,
        "activation": "relu",
        "reconstruction_activation": "sigmoid",
        "optimizer": {"Adam": {"learning_rate": 0.001}},
        "loss": {"MeanSquaredError": {}},
        "epochs": 2
    }

    model = VAEFactory.apply(config)
    model.build(input_shape=(None, 784))
    model.compile(
            loss=get_obj(tf.keras.losses, config["loss"]),
            optimizer=get_obj(tf.keras.optimizers, config["optimizer"])
    )

    for _ in range(config["epochs"]):
        for x, y in train_ds:
            model.train_on_batch(x, y)

        print("{model} epoch metrics: {metrics}".format(
                model=model.name,
                metrics=model.get_metrics_result()
        ))


if __name__ == "__main__":

    import sys

    for model in sys.argv[1:]:
        eval(model)()
