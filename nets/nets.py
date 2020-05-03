
import tensorflow as tf

from nets.layers import *


class Sequential(tf.keras.Model):
    """
    Generic call for sequential models. Includes a
    training flag in the trace.
    """
    def __init__(self):
        super(Sequential, self).__init__()
        self._model_layers = []

    def call(self, inputs, training=None):
        x = inputs
        for lyr in self._model_layers:
            x = lyr(x, training)
        return x


class MLP(Sequential):
    """
    Fully connected model with a configurable number of hidden layers
    and output units.
    """
    def __init__(self, config):

        super(MLP, self).__init__()

        self._config = config
        self._dims = self._config["dense_dims"]
        self._activation = self._config["dense_activation"]
        self._output_dim = self._config["output_dim"]

        self._model_layers = [
            DenseBlock(dims=self._dims, activation=self._activation),
            Softmax(dims=self._output_dim)
        ]


class BasicCNN(Sequential):
    """
    CNN. Allows a configurable number of convolutional layers followed by
     a configurable dense block before the final output.
    """
    def __init__(self, config):

        super(BasicCNN, self).__init__()

        self._config = config
        self._conv_dims = self._config["filters"]
        self._conv_activation = self._config["conv_activation"]
        self._stride = self._config["stride"]
        self._pool = self._config["pool"]
        self._dense_dims = self._config["dense_dims"]
        self._dense_activation = self._config["dense_activation"]
        self._output_dim = self._config["output_dim"]

        self._model_layers = [
            CNNBlock(self._conv_dims, activation=self._conv_activation,
                     stride=self._stride, pool=self._pool),
            DenseBlock(dims=self._dense_dims, activation=self._dense_activation),
            Softmax(dims=self._output_dim)
        ]
