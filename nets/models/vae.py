
"""
VAE
"""

import tensorflow as tf

from nets.blocks.dense import DenseSequentialBlockFactory
from nets.layers.dense import DenseGaussianVariationalEncoder
from nets.models.base import BaseModel
from nets.utils import tf_shape_to_list


@tf.keras.utils.register_keras_serializable("nets")
class VAE(BaseModel):
    """
    Variational autoencoder with dense encoding/decoding layers.
    """

    def __init__(self, encoding_dims, latent_dim, input_dim=None,
                 activation="relu", activity_regularizer=None,
                 reconstruction_activation=None, sparse_flag=False,
                 name="VAE", **kwargs):

        super(VAE, self).__init__(name=name,  **kwargs)

        self._encoding_dims = encoding_dims
        self._latent_dim = latent_dim
        self._input_dim = input_dim
        self._activation = activation
        self._activity_regularizer = activity_regularizer
        self._reconstruction_activation = reconstruction_activation
        self._sparse_flag = sparse_flag

        if self._input_dim is not None:
            self.build(input_shape=self._input_dim)

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

    def build(self, input_shape):
        """
        Build model graph.

        :param input_shape: Dimension of input tensor (not including batch dim)
        """

        # Encoder model with two dense output heads defined above
        self._encoder = DenseGaussianVariationalEncoder(
            encoding_dims=self._encoding_dims,
            latent_dim=self._latent_dim,
            activation=self._activation,
            activity_regularizer=self._activity_regularizer,
            sparse_flag=self._sparse_flag
        )
        self._encoder.build(input_shape=input_shape)
        self._decoder = DenseSequentialBlockFactory.apply(
            hidden =self._encoding_dims[::-1],
            input_shape=(self._latent_dim,),
            activation=self._activation,
            activity_regularizer=self._activity_regularizer
        )
        # Add output layer to reconstruct input onto decoder block
        self._decoder.add(
                tf.keras.layers.Dense(
                        units=tf_shape_to_list(input_shape)[-1],
                        activation=self._reconstruction_activation
                )
        )

    def train_step(self, data):
        """

        :param data:
        :return:
        """

        with tf.GradientTape() as tape:

            z_mean, z_log_var, z = self._encoder(data)
            reconstruction = self._decoder(z)
            reconstruction_loss = self.compiled_loss(data, reconstruction)
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
        _, _, latent = self._encoder(inputs)
        return latent

    def get_config(self):
        config = super(VAE, self).get_config()
        config.update({
            "encoding_dims": self._encoding_dims,
            "latent_dim": self._latent_dim,
            "input_dim": self._input_dim,
            "activation": self._activation,
            "reconstruction_activation": self._reconstruction_activation,
            "activity_regularizer": self._activity_regularizer,
            "sparse_flag": self._sparse_flag
        })
        return config