
"""
VAE
"""

import tensorflow as tf
from nets.layers.dense import DenseBlock
from nets.models.base import BaseModel

from nets.layers.sampling import GaussianSampling


@tf.keras.utils.register_keras_serializable("nets")
class DenseGaussianVariationalEncoder(BaseModel):

    def __init__(self, encoding_dims, latent_dim, activation="relu",
                 activity_regularizer=None, sparse_flag=False, name="VAE",
                 **kwargs):

        super().__init__(name=name, **kwargs)

        self._encoding_dims = encoding_dims
        self._latent_dim = latent_dim
        self._activation = activation
        self._a_regularizer = activity_regularizer
        self._sparse_flag = sparse_flag

        self._input_layer = None
        self._encoding_block = DenseBlock(
                hidden_dims=self._encoding_dims,
                activation=self._activation,
                activity_regularizer=self._activity_regularizer
        )
        self._latent_mean = tf.keras.layers.Dense(self._latent_dim)
        self._latent_log_var = tf.keras.layers.Dense(self._latent_dim)
        self._sampling = GaussianSampling()

    def build(self, input_shape):
        """
        Build portion of model graph that depends on the input shape.
        Also make a call to the parent's build method.

        :param input_shape: Dimension of input tensor (not including batch dim)
        """

        self._input_layer = tf.keras.layers.InputLayer(
                input_shape=(input_shape[-1],)
        )
        super().build(input_shape)

    def call(self, inputs, training=False):
        """

        :param inputs:
        :return:
        """
        x = self._encoding_block(self._input_layer(inputs))
        z_mean = self._latent_mean(x)
        z_log_var = self._latent_log_var(x)
        z = self._sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z

    def get_config(self):
        config = super(DenseGaussianVariationalEncoder, self).get_config()
        config.update({
            "encoding_dims": self._encoding_dims,
            "latent_dim": self._latent_dim,
            "activation": self._activation,
            "activity_regularizer": self._activity_regularizer,
            "sparse_flag": self._sparse_flag
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable("nets")
class DenseGaussianVariationalDecoder(BaseModel):

    def __init__(self, decoding_dims, output_dim, activation="relu",
                 activity_regularizer=None, reconstruction_activation="linear",
                 sparse_flag=False, name="VAE", **kwargs):

        super().__init__(name=name, **kwargs)

        self._decoding_dims = decoding_dims
        self._output_dim = output_dim
        self._activation = activation
        self._a_regularizer = activity_regularizer
        self._reconstruction_activation = reconstruction_activation
        self._sparse_flag = sparse_flag

        self._input_layer = None
        self._decoding_block = DenseBlock(
                hidden_dims=self._decoding_dims,
                activation=self._activation,
                activity_regularizer=self._activity_regularizer
        )
        self._output_layer = tf.keras.layers.Dense(
            units=output_dim, activation=reconstruction_activation
        )

    def build(self, input_shape):
        """
        Build portion of model graph that depends on the input shape.
        Also make a call to the parent's build method.

        :param input_shape: Dimension of input tensor (not including batch dim)
        """

        self._input_layer = tf.keras.layers.InputLayer(
                input_shape=(input_shape[-1],)
        )
        super().build(input_shape)

    def call(self, inputs, training=False):
        """

        :param inputs:
        :return:
        """
        return self._output_layer(self._decoding_block(self._input_layer(inputs)))

    def get_config(self):
        config = super(DenseGaussianVariationalDecoder, self).get_config()
        config.update({
            "decoding_dims": self._decoding_dims,
            "output_dim": self._output_dim,
            "activation": self._activation,
            "activity_regularizer": self._activity_regularizer,
            "reconstruction_activation": self._reconstruction_activation,
            "sparse_flag": self._sparse_flag
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable("nets")
class VAE(BaseModel):
    """
    Variational autoencoder with dense encoding/decoding layers.
    """

    def __init__(self, encoding_dims, latent_dim, input_shape=None,
                 activation="relu", activity_regularizer=None,
                 reconstruction_activation=None, sparse_flag=False,
                 name="VAE", **kwargs):

        super().__init__(name=name,  **kwargs)

        self._encoding_dims = encoding_dims
        self._latent_dim = latent_dim
        self._input_shape = input_shape
        self._activation = activation
        self._activity_regularizer = activity_regularizer
        self._reconstruction_activation = reconstruction_activation
        self._sparse_flag = sparse_flag

        self._loss_tracker = tf.keras.metrics.Mean(
                name="loss"
        )
        self._reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self._discrepancy_loss_tracker =tf. keras.metrics.Mean(
                name="discrepancy_loss"
        )

        self._tracked_metrics = [
            self._loss_tracker,
            self._reconstruction_loss_tracker,
            self._discrepancy_loss_tracker
        ]

        # Encoding layer
        self._encoder = DenseGaussianVariationalEncoder(
            encoding_dims=self._encoding_dims,
            latent_dim=self._latent_dim,
            activation=self._activation,
            activity_regularizer=self._activity_regularizer
        )

        # Input layer and decode blocks can only be defined now if we
        # received an input_shape. Otherwise deferred to a .build()
        # call by the client.
        self._decoder = None
        if self._input_shape is not None:
            self.build(self._input_shape)

    def build(self, input_shape):
        """
        Build portion of model graph that depends on the input shape.
        Also make a call to the parent's build method.

        :param input_shape: Dimension of input tensor
        """
        self._input_shape = input_shape
        self._input_dim = input_shape[-1]
        self._encoder.build(self._input_shape)

        self._decoder = DenseGaussianVariationalDecoder(
            decoding_dims=self._encoding_dims[::-1],
            output_dim=self._input_shape[-1],
            activation=self._activation,
            activity_regularizer=self._activity_regularizer,
            reconstruction_activation=self._reconstruction_activation
        )
        self._decoder.build((None, self._latent_dim))

        super().build(self._input_shape)

    def train_step(self, data):
        """

        :param data:
        :return:
        """

        x, y = data
        with tf.GradientTape() as tape:

            z_mean, z_log_var, z = self._encoder.__call__(x)
            reconstruction = self._decoder.__call__(z)

            reconstruction_loss = self.compiled_loss(
                    tf.reshape(x, (-1, self._input_shape[-1])), reconstruction
            )
            discrepancy_loss = -0.5 * (
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            )
            discrepancy_loss = tf.reduce_mean(tf.reduce_sum(discrepancy_loss))
            total_loss = reconstruction_loss + discrepancy_loss

        gradients = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))

        self._loss_tracker.update_state(total_loss)
        self._reconstruction_loss_tracker.update_state(reconstruction_loss)
        self._discrepancy_loss_tracker.update_state(discrepancy_loss)

        return {m.name: m.result() for m in self._tracked_metrics}

    def encode(self, inputs):
        """
        Return sampled latent values.
        :param inputs:
        :return:
        """
        _, _, latent = self._encoder.__call__(inputs)
        return latent

    def call(self, inputs, training=False):
        """
        Return reconstructed inputs.
        :param inputs:
        :return:
        """
        z_mean, z_log_var, z = self._encoder.__call__(inputs)
        return self._decoder.__call__(z)

    def get_config(self):
        config = super(VAE, self).get_config()
        config.update({
            "encoding_dims": self._encoding_dims,
            "latent_dim": self._latent_dim,
            "activation": self._activation,
            "reconstruction_activation": self._reconstruction_activation,
            "activity_regularizer": self._activity_regularizer,
            "sparse_flag": self._sparse_flag
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @property
    def encoder(self):
        return self._encoder

    @property
    def decoder(self):
        return self._decoder