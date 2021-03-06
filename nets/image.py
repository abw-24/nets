
from tensorflow.keras import Model as BaseModel
from tensorflow.keras.layers import Flatten

from nets.layers import *


class CNN(BaseModel):
    """
    CNN. Allows a configurable number of convolutional layers via the CNNBlock,
    followed by a configurable dense block before a final output layer.
    """
    def __init__(self, config):

        super(CNN, self).__init__()

        self._config = config
        self._n_filters = self._config["conv_filters"]
        self._kernel = self._config.get("conv_kernel", 3)
        self._conv_activation = self._config.get("conv_activation", "relu")
        self._stride = self._config.get("stride", (1,1))
        self._padding = self._config.get("padding", "same")
        self._pool = self._config.get("pool", True)
        self._dense_dims = self._config["dense_dims"]
        self._dense_activation = self._config.get("dense_activation", "relu")
        self._output_dim = self._config["output_dim"]
        self._output_activation = self._config.get("output_activation", "softmax")

        self._model_layers = {
            "0": ConvBlock(
                    self._n_filters,
                    kernel=self._kernel,
                    stride=self._stride,
                    padding=self._padding,
                    activation=self._conv_activation,
                    pool=self._pool
            ),
            "1": Flatten(),
            "2": DenseBlock(dims=self._dense_dims, activation=self._dense_activation),
            "3": DenseBlock(dims=[self._output_dim], activation=self._output_activation)
        }

    def call(self, inputs, training=None):
        x = inputs
        for i in range(len(self._model_layers)):
            x = self._model_layers[str(i)](x, training=training)
        return x

    def get_config(self):
        config = super(CNN, self).get_config()
        config.update({"config": self._config})
        return config


class ResNet(BaseModel):
    """
    ResNet CNN. Allows a configurable number of initial convolutional layers
    via the CNNBlock, followed by a configurable ResidualBlock, composed of
     a set of residual or multi-path residual block convolutional layers.
    """
    def __init__(self, config):

        super(ResNet, self).__init__()

        self._config = config

        self._conv_filters = self._config["conv_filters"]
        self._conv_kernel = self._config.get("conv_kernel", 3)
        self._conv_activation = self._config.get("conv_activation", "relu")
        self._stride = self._config.get("stride", (1,1))
        self._padding = self._config.get("padding", "same")
        self._pool = self._config.get("pool", True)

        self._res_depth = self._config["res_depth"]
        self._n_paths = self._config.get("res_paths", None)
        self._res_filters = self._config["res_filters"]
        self._res_activation = self._config.get("res_activation", "relu")

        self._dense_dims = self._config["dense_dims"]
        self._dense_activation = self._config.get("dense_activation", "relu")
        self._output_dim = self._config["output_dim"]
        self._output_activation = self._config.get("output_activation", "softmax")

        if self._n_paths is None:
            res_block = ResidualBlock(
                    block_depth=self._res_depth,
                    filters=self._res_filters,
                    activation=self._res_activation
            )
        else:
            res_block = MultiPathResidualBlock(
                    block_depth=self._res_depth,
                    n_paths=self._n_paths,
                    filters=self._res_filters,
                    activation=self._res_activation
            )

        self._model_layers = {
            "0": ConvBlock(
                    self._conv_filters,
                    kernel=self._conv_kernel,
                    stride=self._stride,
                    padding=self._padding,
                    activation=self._conv_activation,
                    pool=self._pool
            ),
            "1": res_block,
            "2": Flatten(),
            "3": DenseBlock(dims=self._dense_dims, activation=self._dense_activation),
            "4": DenseBlock(dims=[self._output_dim], activation=self._output_activation)
        }

    def call(self, inputs, training=None):
        x = inputs
        for i in range(len(self._model_layers)):
            x = self._model_layers[str(i)](x, training=training)
        return x

    def get_config(self):
        config = super(ResNet, self).get_config()
        config.update({"config": self._config})
        return config
