
import tensorflow as tf
from tensorflow import keras


def smoke_decorator(fn):
    """
    Wrap network tests to catch and print exceptions.
    :param fn: Test method
    :return: Method wrapped with try/except
    """
    def _inner():
        try:
            fn()
        except BaseException() as e:
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
    import nets.nets as nets
    import nets.train as t

    # load mnist data, flatten, and normalize to 0-1
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape((x_train.shape[0], 784))
    x_train = x_train / 255.0

    # create a batch feed from the train tensors
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)) \
        .shuffle(10000) \
        .batch(32)

    config = {
        "dense_dims": [32],
        "dense_activation": "relu",
        "output_dim": 10,
        "output_activation": "softmax",
        "optimizer": {"Adam": {"learning_rate": 0.001}},
        "loss": {"SparseCategoricalCrossentropy": {}},
        "epochs": 3
    }

    compiled_model = t.model_init(nets.MLP(config), config, (None, 784))

    for _ in range(config["epochs"]):
        loss = 0.0
        for x, y in train_ds:
            loss, grads = t.grad(compiled_model, x, y)
            updates = zip(grads, compiled_model.trainable_variables)
            compiled_model.optimizer.apply_gradients(updates)
        print("Epoch loss: {loss}".format(loss=loss))


@smoke_decorator
def basic_cnn():
    """
    Test mlp on mnist data.
    """
    import nets.nets as nets
    import nets.train as t

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
        "epochs": 3
    }

    compiled_model = t.model_init(nets.BasicCNN(config), config, (None, 28, 28, 1))

    for _ in range(config["epochs"]):
        loss = 0.0
        step = 0
        for x, y in train_ds:
            loss_, grads = t.grad(compiled_model, x, y)
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
    import nets.nets as nets
    import nets.train as t

    # load mnist data, reshape to have channel dim, and normalize to 0-1
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
    x_train = x_train / 255.0

    # create a batch feed from the train tensors
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)) \
        .shuffle(10000) \
        .batch(32)

    config = {
        "conv_filters": [16],
        "res_filters": [[32, 32], [64, 64]],
        "res_depth": 2,
        "dense_dims": [32],
        "output_dim": 10,
        "output_activation": "softmax",
        "optimizer": {"Adam": {"learning_rate": 0.001}},
        "loss": {"SparseCategoricalCrossentropy": {}},
        "epochs": 10
    }

    compiled_model = t.model_init(nets.ResNet(config), config, (None, 28, 28, 1))

    for _ in range(config["epochs"]):
        loss = 0.0
        step = 0
        for x, y in train_ds:
            loss_, grads = t.grad(compiled_model, x, y)
            updates = zip(grads, compiled_model.trainable_variables)
            compiled_model.optimizer.apply_gradients(updates)
            loss += loss_
            step += 1
        print("Epoch loss: {loss}".format(loss=loss/float(step)))


if __name__ == "__main__":

    tf.keras.backend.set_floatx('float64')

    tests = [resnet]

    for t in tests:
        status = "passed!" if t() else "failed!"
        print("{f} status: {s}".format(f=str(t), s=status))
