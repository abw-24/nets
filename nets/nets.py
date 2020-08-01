
from tensorflow.keras import Model as BaseModel
from tensorflow.keras.layers import Flatten

from nets.layers import *


class MLP(BaseModel):
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
            x = lyr(x, training=training)
        return x

    def get_config(self):
        config = super(MLP, self).get_config()
        config.update({"config": self._config})
        return config


class BasicCNN(BaseModel):
    """
    CNN. Allows a configurable number of convolutional layers via the CNNBlock,
    followed by a configurable dense block before a final output layer.
    """
    def __init__(self, config):

        super(BasicCNN, self).__init__()

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

        self._model_layers = [
            ConvBlock(
                    self._n_filters,
                    kernel=self._kernel,
                    stride=self._stride,
                    padding=self._padding,
                    activation=self._conv_activation,
                    pool=self._pool
            ),
            Flatten(),
            DenseBlock(dims=self._dense_dims, activation=self._dense_activation),
            DenseBlock(dims=[self._output_dim], activation=self._output_activation)
        ]

    def call(self, inputs, training=False):
        x = inputs
        for lyr in self._model_layers:
            x = lyr(x, training=training)
        return x

    def get_config(self):
        config = super(BasicCNN, self).get_config()
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
            resblock = ResidualBlock(
                    block_depth=self._res_depth,
                    filters=self._res_filters,
                    activation=self._res_activation
            )
        else:
            resblock = MultiPathResidualBlock(
                    block_depth=self._res_depth,
                    n_paths=self._n_paths,
                    filters=self._res_filters,
                    activation=self._res_activation
            )

        self._model_layers = [
            ConvBlock(
                    self._conv_filters,
                    kernel=self._conv_kernel,
                    stride=self._stride,
                    padding=self._padding,
                    activation=self._conv_activation,
                    pool=self._pool
            ),
            resblock,
            Flatten(),
            DenseBlock(dims=self._dense_dims, activation=self._dense_activation),
            DenseBlock(dims=[self._output_dim], activation=self._output_activation)
        ]

    def call(self, inputs, training=False):
        x = inputs
        for lyr in self._model_layers:
            x = lyr(x, training=training)
        return x

    def get_config(self):
        config = super(ResNet, self).get_config()
        config.update({"config": self._config})
        return config


class DenseVAE(BaseModel):
    """
    Variational autoencoder with dense embedding connections.
    """

    def __init__(self, config, **kwargs):

        super(DenseVAE, self).__init__(name="VAE", **kwargs)

        self._config = config
        self._input_dim = self._config["input_dim"]
        self._encoding_dims = self._config["encoding_dims"]
        self._latent_dim = self._config["latent_dim"]
        self._activation = self._config["activation"]

        self._encoder = DenseEncoder(
                mapping_dims=self._encoding_dims,
                latent_dim=self._latent_dim,
                activation=self._activation
        )
        self._decoder = DenseDecoder(
                inverse_mapping_dims=self._encoding_dims[::-1],
                input_dim=self._input_dim,
                activation=self._activation
        )

    def call(self, inputs, training=False):
        z_mean, z_log_var, z = self._encoder(inputs)
        reconstructed = self._decoder(z)
        # add kl loss as a zero arg lambda function to make it callable.
        # this penalization is what enforces normality
        kl_loss = lambda: -0.5 * tf.reduce_mean(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
        )
        self.add_loss(kl_loss)
        return reconstructed

    def get_config(self):
        config = super(ResNet, self).get_config()
        config.update({"config": self._config})
        return config
