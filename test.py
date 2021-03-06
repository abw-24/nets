
import tensorflow as tf
from tensorflow import keras

import nets.train as train


def smoke_decorator(fn):
    """
    Wrap network tests to catch and print exceptions.
    :param fn: Test method
    :return: Method wrapped with try/except
    """
    def _inner():
        try:
            fn()
        except Exception as e:
            print(e)
            return False
        else:
            return True
    return _inner


@smoke_decorator
def mlp():
    """
    Test mlp on mnist data.
    """
    import nets.dense as dense

    # load mnist data, flatten, and normalize to 0-1
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape((x_train.shape[0], 784))
    x_train = x_train / 255.0

    # create a batch feed from the train tensors
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)) \
        .shuffle(10000) \
        .batch(32)

    config = {
        "dense_dims": [32, 16],
        "dense_activation": "relu",
        "output_dim": 10,
        "output_activation": "softmax",
        "optimizer": {"Adam": {"learning_rate": 0.001}},
        "loss": {"SparseCategoricalCrossentropy": {}},
        "epochs": 2
    }

    compiled_model = train.model_init(
            dense.MLP(config),
            config["loss"],
            config["optimizer"],
            (None, 784)
    )

    for _ in range(config["epochs"]):
        loss = 0.0
        step = 0
        for x, y in train_ds:
            loss_, grads = train.grad(compiled_model, x, y)
            updates = zip(grads, compiled_model.trainable_variables)
            compiled_model.optimizer.apply_gradients(updates)
            loss += loss_
            step += 1
        print("Epoch loss: {loss}".format(loss=loss/float(step)))


@smoke_decorator
def basic_cnn():
    """
    Test mlp on mnist data.
    """
    import nets.image as img

    # load mnist data, reshape to have channel dim, and normalize to 0-1
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
    x_train = x_train / 255.0

    # create a batch feed from the train tensors
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)) \
        .shuffle(10000) \
        .batch(32)

    config = {
        "conv_filters": [16, 32],
        "dense_dims": [32],
        "output_dim": 10,
        "output_activation": "softmax",
        "optimizer": {"Adam": {"learning_rate": 0.001}},
        "loss": {"SparseCategoricalCrossentropy": {}},
        "epochs": 2
    }

    compiled_model = train.model_init(
            img.CNN(config),
            config["loss"],
            config["optimizer"],
            (None, 28, 28, 1)
    )

    for _ in range(config["epochs"]):
        loss = 0.0
        step = 0
        for x, y in train_ds:
            loss_, grads = train.grad(compiled_model, x, y)
            updates = zip(grads, compiled_model.trainable_variables)
            compiled_model.optimizer.apply_gradients(updates)
            loss += loss_
            step += 1
        print("Epoch loss: {loss}".format(loss=loss/float(step)))


@smoke_decorator
def resnet():
    """
    Test mlp on mnist data.
    """
    import nets.image as img

    # load mnist data, reshape to have channel dim, and normalize to 0-1
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
    x_train = x_train / 255.0

    # create a batch feed from the train tensors
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)) \
        .shuffle(10000) \
        .batch(32)

    config = {
        "conv_filters": [24],
        "res_filters": [8, 24],
        "res_depth": 1,
        "dense_dims": [24],
        "output_dim": 10,
        "output_activation": "softmax",
        "optimizer": {"Adam": {"learning_rate": 0.001}},
        "loss": {"SparseCategoricalCrossentropy": {}},
        "epochs": 2
    }

    compiled_model = train.model_init(
            img.ResNet(config),
            config["loss"],
            config["optimizer"],
            (None, 28, 28, 1)
    )

    for _ in range(config["epochs"]):
        loss = 0.0
        step = 0
        for x, y in train_ds:
            loss_, grads = train.grad(compiled_model, x, y)
            updates = zip(grads, compiled_model.trainable_variables)
            compiled_model.optimizer.apply_gradients(updates)
            loss += loss_
            step += 1
        print("Epoch loss: {loss}".format(loss=loss/float(step)))


@smoke_decorator
def dense_vae():
    """
    Test mlp on mnist data.
    """
    import nets.dense as dense

    # load mnist data, flatten, and normalize to 0-1
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape((x_train.shape[0], 784))
    x_train = x_train / 255.0

    # create a batch feed from the train tensors
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)) \
        .shuffle(10000) \
        .batch(32)

    config = {
        "encoding_dims": [64, 32],
        "latent_dim": 10,
        "activation": "relu",
        "optimizer": {"Adam": {"learning_rate": 0.001}},
        "loss": {"MeanAbsoluteError": {}},
        "epochs": 2,
        "sparse_flag": False
    }

    compiled_model = train.model_init(
            dense.DenseVAE(config),
            config["loss"],
            config["optimizer"],
            (None, 784)
    )

    for _ in range(config["epochs"]):
        loss = 0.0
        step = 0
        for x, y in train_ds:
            loss_, grads = train.grad(compiled_model, x, x)
            updates = zip(grads, compiled_model.trainable_variables)
            compiled_model.optimizer.apply_gradients(updates)
            loss += loss_
            step += 1
        print("Epoch loss: {loss}".format(loss=loss/float(step)))


@smoke_decorator
def dense_ae():
    """
    Test mlp on mnist data.
    """
    import nets.dense as dense

    # load mnist data, flatten, and normalize to 0-1
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape((x_train.shape[0], 784))
    x_train = x_train / 255.0

    # create a batch feed from the train tensors
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)) \
        .shuffle(10000) \
        .batch(32)

    config = {
        "encoding_dims": [64, 32],
        "latent_dim": 10,
        "activation": "relu",
        "optimizer": {"Adam": {"learning_rate": 0.001}},
        "loss": {"MeanAbsoluteError": {}},
        "activity_regularizer": {"l1_l2": {"l1": 0.0075, "l2": 0.0025}},
        "epochs": 2,
        "sparse_flag": False
    }

    compiled_model = train.model_init(
            dense.DenseAE(config),
            config["loss"],
            config["optimizer"],
            (None, 784)
    )

    for _ in range(config["epochs"]):
        loss = 0.0
        step = 0
        for x, y in train_ds:
            loss_, grads = train.grad(compiled_model, x, x)
            updates = zip(grads, compiled_model.trainable_variables)
            compiled_model.optimizer.apply_gradients(updates)
            loss += loss_
            step += 1
        print("Epoch loss: {loss}".format(loss=loss/float(step)))


if __name__ == "__main__":

    tf.keras.backend.set_floatx('float64')

    tests = ["dense_ae"]

    for t in tests:
        status = "passed!" if eval(t)() else "failed!"
        print("{f} status: {s}".format(f=t, s=status))
