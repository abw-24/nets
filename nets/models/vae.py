
"""
VAE
"""

import tensorflow as tf

from nets.layers.dense import DenseBlock, DenseGaussianVariationalEncoder

from nets.models.base import BaseModel


@tf.keras.utils.register_keras_serializable("nets")
class VAE(BaseModel):
    """
    Variational autoencoder with dense encoding/decoding layers.
    """

    def __init__(self, encoding_dims, latent_dim, input_shape=None,
                 activation="relu", activity_regularizer=None,
                 reconstruction_activation=None, sparse_flag=False,
                 name="VAE", **kwargs):

        super(VAE, self).__init__(name=name,  **kwargs)

        self._encoding_dims = encoding_dims
        self._latent_dim = latent_dim
        self._input_shape = input_shape
        self._activation = activation
        self._activity_regularizer = activity_regularizer
        self._reconstruction_activation = reconstruction_activation
        self._sparse_flag = sparse_flag

        self._total_loss_tracker = tf.keras.metrics.Mean(
                name="total_loss"
        )
        self._reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self._discrepancy_loss_tracker =tf. keras.metrics.Mean(
                name="discrepancy_loss"
        )

        self._tracked_metrics = [
            self._total_loss_tracker,
            self._reconstruction_loss_tracker,
            self._discrepancy_loss_tracker
        ]

        # Input layer and decode blocks can only be defined now if we
        # received an input_shape. Otherwise deferred to a .build()
        # call by the client.
        self._input_layer = None
        self._decode_block = None
        self._output_layer = None
        self._input_dim = None
        if self._input_shape is not None:
            self.build(self._input_shape)

        # Encoding layer
        self._encoder = DenseGaussianVariationalEncoder(
            encoding_dims=self._encoding_dims,
            latent_dim=self._latent_dim,
            activation=self._activation,
            activity_regularizer=self._activity_regularizer
        )

    def build(self, input_shape):
        """
        Build portion of model graph that depends on the input shape.

        :param input_shape: Dimension of input tensor (not including batch dim)
        """
        self._input_layer = tf.keras.layers.InputLayer(
                input_shape=(input_shape[-1],)
        )
        self._decode_block = DenseBlock(
                hidden=self._encoding_dims,
                activation=self._activation,
                activity_regularizer=self._activity_regularizer
        )
        self._output_layer = tf.keras.layers.Dense(
                units=input_shape[-1], activation=self._reconstruction_activation
        )
        self._input_dim = tf.constant(input_shape[-1])

        super().build(input_shape)

    def train_step(self, data):
        """

        :param data:
        :return:
        """

        x, y = data
        with tf.GradientTape() as tape:

            z_mean, z_log_var, z = self._encoder(self._input_layer(x))
            reconstruction = self._output_layer(self._decode_block(z))

            reconstruction_loss = self.compiled_loss(
                    tf.reshape(x, (-1, self._input_dim)), reconstruction
            )
            discrepancy_loss = -0.5 * (
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            )
            discrepancy_loss = tf.reduce_mean(tf.reduce_sum(discrepancy_loss))
            total_loss = reconstruction_loss + discrepancy_loss

        gradients = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))

        self._total_loss_tracker.update_state(total_loss)
        self._reconstruction_loss_tracker.update_state(reconstruction_loss)
        self._discrepancy_loss_tracker.update_state(discrepancy_loss)

        return {m.name: m.result() for m in self._tracked_metrics}

    def encode(self, inputs):
        """
        Return sampled latent values.
        :param inputs:
        :return:
        """
        _, _, latent = self._encoder(self._input_layer(inputs))
        return latent

    def call(self, inputs, training=False):
        """
        Return sampled latent values.
        :param inputs:
        :return:
        """
        _, _, latent = self._encoder(self._input_layer(inputs))
        return self._output_layer(self._decode_block(latent))

    def get_config(self):
        config = super(VAE, self).get_config()
        config.update({
            "encoding_dims": self._encoding_dims,
            "latent_dim": self._latent_dim,
            "input_dim": self._input_shape,
            "activation": self._activation,
            "reconstruction_activation": self._reconstruction_activation,
            "activity_regularizer": self._activity_regularizer,
            "sparse_flag": self._sparse_flag
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)