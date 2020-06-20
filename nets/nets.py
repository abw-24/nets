
import tensorflow as tf

from nets.layers import *


class MLP(tf.keras.Model):
    """
    Fully connected model with a configurable number of hidden layers
    and output units
    """
    def __init__(self, config):

        super(MLP, self).__init__()

        self._config = config
        self._dims = self._config["dense_dims"]
        self._output_dim = self._config["output_dim"]
        self._activation = self._config.get("dense_activation", "relu")
        self._output_activation = self._config.get("output_activation", "softmax")

        self._model_layers = [
            DenseBlock(dims=self._dims, activation=self._activation),
            DenseBlock(dims=[self._output_dim], activation=self._output_activation)
        ]

    def call(self, inputs, training=None):
        x = inputs
        for lyr in self._model_layers:
            x = lyr(x, training)
        return x


class BasicCNN(tf.keras.Model):
    """
    CNN. Allows a configurable number of convolutional layers via the CNNBlock,
    followed by a configurable dense block before the final output layer.
    """
    def __init__(self, config):

        super(BasicCNN, self).__init__()

        self._config = config
        self._n_filters = self._config["filters"]
        self._kernel = self._config.get("kernel", 3)
        self._conv_activation = self._config.get("conv_activation", "relu")
        self._stride = self._config.get("stride", (1,1))
        self._pool = self._config.get("pool", True)
        self._dense_dims = self._config["dense_dims"]
        self._dense_activation = self._config.get("dense_activation", "relu")
        self._output_dim = self._config["output_dim"]
        self._output_activation = self._config.get("output_activation", "softmax")

        self._model_layers = [
            CNNBlock(self._n_filters, self._kernel, self._stride, self._conv_activation, self._pool),
            DenseBlock(dims=self._dense_dims, activation=self._dense_activation),
            DenseBlock(dims=[self._output_dim], activation=self._output_activation)
        ]

    def call(self, inputs, training=None):
        x = inputs
        for lyr in self._model_layers:
            x = lyr(x, training)
        return x