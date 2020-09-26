
from tensorflow.keras import Model as BaseModel

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

        self._model_layers = {
            "0": DenseBlock(dims=self._dims, activation=self._activation),
            "1": DenseBlock(dims=[self._output_dim], activation=self._output_activation)
        }

    def call(self, inputs, training=None):
        x = inputs
        for i in range(len(self._model_layers)):
            x = self._model_layers[str(i)](x, training=training)
        return x

    def get_config(self):
        config = super(MLP, self).get_config()
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
        self._hidden_representation = None

    def call(self, inputs, training=False):
        z_mean, z_log_var, self._hidden_representation = self._encoder(inputs)
        reconstructed = self._decoder(self._hidden_representation)
        # add kl loss as a zero arg lambda function to make it callable.
        # this penalization is what enforces normality
        if training:
            kl_loss = lambda: -0.5 * tf.reduce_mean(
                z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
            )
            self.add_loss(kl_loss)
        return reconstructed

    def get_config(self):
        config = super(DenseVAE, self).get_config()
        config.update({"config": self._config})
        return config

    @property
    def hidden(self):
        return self._hidden_representation