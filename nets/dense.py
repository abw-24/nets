
from tensorflow.keras import Model as BaseModel

from nets.layers import *
from nets.utils import get_tf


class MLP(BaseModel):
    """
    Fully connected model with a configurable number of hidden layers
    and output units
    """
    def __init__(self, config, **kwargs):

        super(MLP, self).__init__(name="MLP", **kwargs)

        self._config = config
        self._dims = self._config["dense_dims"]
        self._output_dim = self._config["output_dim"]
        self._activation = self._config.get("dense_activation", "relu")
        self._output_activation = self._config.get("output_activation", "softmax")
        self._k_regularizer = self._config.get("kernel_regularizer", None)
        self._a_regularizer = self._config.get("activity_regularizer", None)

        self._model_layers = {
            "0": DenseBlock(
                    dims=self._dims,
                    activation=self._activation,
                    kernel_regularizer=get_tf(tf.keras.regularizers, self._k_regularizer),
                    activity_regularizer=get_tf(tf.keras.regularizers, self._a_regularizer)
            ),
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

        self._encoding_dims = config["encoding_dims"]
        self._latent_dim = config["latent_dim"]
        self._activation = config["activation"]
        self._sparse_flag = config.get("sparse_flag", False)
        self._a_regularizer = config.get("activity_regularizer", None)

    def build(self, input_shape):

        self._input_layer = tf.keras.layers.InputLayer(
                input_shape=(input_shape[-1],),
                sparse=self._sparse_flag
        )

        self._encoder = DenseVariationalEncoder(
                mapping_dims=self._encoding_dims,
                latent_dim=self._latent_dim,
                activation=self._activation,
                activity_regularizer=get_tf(tf.keras.regularizers, self._a_regularizer)
        )
        self._decoder = DenseVariationalDecoder(
                inverse_mapping_dims=self._encoding_dims[::-1],
                input_dim=input_shape[-1],
                activation=self._activation,
                activity_regularizer=get_tf(tf.keras.regularizers, self._a_regularizer)
        )

    def call(self, inputs, training=False):
        init_inputs = self._input_layer(inputs)
        z_mean, z_log_var, hidden = self._encoder(init_inputs)
        reconstructed = self._decoder(hidden)
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
        config.update({
            "encoding_dims": self._encoding_dims,
            "latent_dim": self._latent_dim,
            "activation": self._activation,
            "sparse_flag": self._sparse_flag
        })
        return config

    def encode(self, inputs):
        z_mean, z_log_var, hidden = self._encoder(inputs)
        return hidden


class DenseAE(BaseModel):
    """
    Classic feedforward autoencoder.
    """

    def __init__(self, config, **kwargs):

        super(DenseAE, self).__init__(name="AE", **kwargs)

        self._encoding_dims = config["encoding_dims"]
        self._latent_dim = config["latent_dim"]
        self._activation = config["activation"]
        self._sparse_flag = config.get("sparse_flag", False)
        self._a_regularizer = config.get("activity_regularizer", None)

    def build(self, input_shape):

        self._input_layer = tf.keras.layers.InputLayer(
                input_shape=(input_shape[-1],),
                sparse=self._sparse_flag
        )

        self._encoder = DenseEncoder(
                mapping_dims=self._encoding_dims,
                latent_dim=self._latent_dim,
                activation=self._activation,
                activity_regularizer=get_tf(tf.keras.regularizers, self._a_regularizer)
        )
        self._decoder = DenseDecoder(
                inverse_mapping_dims=self._encoding_dims[::-1],
                input_dim=input_shape[-1],
                activation=self._activation,
                activity_regularizer=get_tf(tf.keras.regularizers, self._a_regularizer)
        )

    def call(self, inputs, training=False):
        init_inputs = self._input_layer(inputs)
        hidden = self._encoder(init_inputs)
        return self._decoder(hidden)

    def get_config(self):
        config = super(DenseAE, self).get_config()
        config.update({
            "encoding_dims": self._encoding_dims,
            "latent_dim": self._latent_dim,
            "activation": self._activation,
            "sparse_flag": self._sparse_flag
        })
        return config

    def encode(self, inputs):
        return self._encoder(inputs)