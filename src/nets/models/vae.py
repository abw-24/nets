
"""
VAE
"""

import tensorflow as tf
from tensorflow_model_remediation.min_diff.losses import MMDLoss

from nets.layers.dense import DenseBlock
from nets.models.base import BaseTFKerasModel
from nets.layers.sampling import GaussianSampling


@tf.keras.utils.register_keras_serializable("nets")
class GaussianDenseVariationalEncoder(BaseTFKerasModel):
    """
    Variational encoder.
    """

    def __init__(self, encoding_dims, latent_dim, activation="relu",
                 activity_regularizer=None, kernel_regularizer=None,
                 spectral_norm=False, name="DGVE", **kwargs):

        super().__init__(name=name, **kwargs)

        self._encoding_dims = encoding_dims
        self._latent_dim = latent_dim
        self._activation = activation
        self._activity_regularizer = activity_regularizer
        self._kernel_regularizer = kernel_regularizer
        self._spectral_norm = spectral_norm

        self._input_layer = None
        self._encode_block = DenseBlock(
                hidden_dims=self._encoding_dims,
                activation=self._activation,
                activity_regularizer=self._activity_regularizer,
                kernel_regularizer=self._kernel_regularizer,
                spectral_norm=self._spectral_norm
        )
        self._latent_mean = tf.keras.layers.Dense(self._latent_dim)
        self._latent_log_var = tf.keras.layers.Dense(self._latent_dim)
        self._sampling = GaussianSampling()

    def build(self, input_shape):
        """
        Build portion of model graph that depends on the input shape
        """

        self._input_layer = tf.keras.layers.InputLayer(
                input_shape=(input_shape[-1],)
        )
        super().build(input_shape)

    def call(self, inputs, training=False):

        dense_block = self._encode_block.__call__(self._input_layer.__call__(inputs))
        latent_mean = self._latent_mean.__call__(dense_block)
        latent_log_var = self._latent_log_var.__call__(dense_block)
        encoding = self._sampling.__call__((latent_mean, latent_log_var))

        return (latent_mean, latent_log_var, encoding)

    def get_config(self):
        config = super(GaussianDenseVariationalEncoder, self).get_config()
        config.update({
            "encoding_dims": self._encoding_dims,
            "latent_dim": self._latent_dim,
            "activation": self._activation,
            "activity_regularizer": self._activity_regularizer,
            "kernel_regularizer": self._kernel_regularizer,
            "spectral_norm": self._spectral_norm
        })
        return config


@tf.keras.utils.register_keras_serializable("nets")
class GaussianDenseVariationalDecoder(BaseTFKerasModel):

    def __init__(self, decoding_dims, output_dim, activation="relu",
                 activity_regularizer=None, kernel_regularizer=None,
                 spectral_norm=False, reconstruction_activation="linear",
                 name="DGVD", **kwargs):

        super().__init__(name=name, **kwargs)

        self._decoding_dims = decoding_dims
        self._output_dim = output_dim
        self._activation = activation
        self._activity_regularizer = activity_regularizer
        self._kernel_regularizer = kernel_regularizer
        self._spectral_norm = spectral_norm
        self._reconstruction_activation = reconstruction_activation

        self._input_layer = None
        self._decode_block = DenseBlock(
                hidden_dims=self._decoding_dims,
                activation=self._activation,
                activity_regularizer=self._activity_regularizer
        )
        self._output_layer = tf.keras.layers.Dense(
            units=output_dim, activation=reconstruction_activation
        )

    def build(self, input_shape):
        """
        Build portion of model graph that depends on the input shape
        """

        self._input_layer = tf.keras.layers.InputLayer(
                input_shape=(input_shape[-1],)
        )
        super().build(input_shape)

    def call(self, inputs, training=False):

        decoded = self._decode_block.__call__(self._input_layer.__call__(inputs))
        return self._output_layer.__call__(decoded)

    def get_config(self):
        config = super(GaussianDenseVariationalDecoder, self).get_config()
        config.update({
            "decoding_dims": self._decoding_dims,
            "output_dim": self._output_dim,
            "activation": self._activation,
            "activity_regularizer": self._activity_regularizer,
            "kernel_regularizer": self._kernel_regularizer,
            "spectral_norm": self._spectral_norm,
            "reconstruction_activation": self._reconstruction_activation
        })
        return config


@tf.keras.utils.register_keras_serializable("nets")
class GaussianDenseVAE(BaseTFKerasModel):
    """
    Variational autoencoder with dense encoding/decoding layers. Defaults
    to Max Mean Discrepancy as the dissimilarity measure per the
    InfoVAE line of research on learning meaningful latent codes.

    Paper: https://arxiv.org/pdf/1706.02262.pdf
    """

    def __init__(self, encoding_dims, latent_dim, input_shape=None,
                 activation="relu", activity_regularizer=None,
                 kernel_regularizer=None, spectral_norm=False,
                 reconstruction_activation="linear", discrepancy_loss="mmd",
                 name="VAE", **kwargs):

        super().__init__(name=name,  **kwargs)

        self._encoding_dims = encoding_dims
        self._latent_dim = latent_dim
        self._input_shape = input_shape
        self._activation = activation
        self._activity_regularizer = activity_regularizer
        self._kernel_regularizer = kernel_regularizer
        self._spectral_norm = spectral_norm
        self._reconstruction_activation = reconstruction_activation
        self._discrepancy_loss = discrepancy_loss.lower()

        self._loss_tracker = tf.keras.metrics.Mean(
                name="loss"
        )
        self._reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self._discrepancy_loss_tracker =tf. keras.metrics.Mean(
                name="discrepancy_loss"
        )
        # For simplicity, create an MMD loss object even if not needed
        self._mmd_loss = MMDLoss(kernel="gaussian")

        self._tracked_metrics = [
            self._loss_tracker,
            self._reconstruction_loss_tracker,
            self._discrepancy_loss_tracker
        ]

        # Encoder
        self._encoder = GaussianDenseVariationalEncoder(
            encoding_dims=self._encoding_dims,
            latent_dim=self._latent_dim,
            activation=self._activation,
            activity_regularizer=self._activity_regularizer,
            kernel_regularizer=self._kernel_regularizer,
            spectral_norm=self._spectral_norm
        )

        # Input layer and decoder can only be defined now if we
        # received an input_shape. Otherwise deferred to a .build()
        # call by the user.
        self._input_dim = None
        self._decoder = None
        if self._input_shape is not None:
            self.build(self._input_shape)

    def build(self, input_shape):
        """
        Build portion of model graph that depends on the input shape
        """
        self._input_shape = input_shape
        self._input_dim = input_shape[-1]

        self._encoder.build(self._input_shape)

        self._decoder = GaussianDenseVariationalDecoder(
            decoding_dims=self._encoding_dims[::-1],
            output_dim=self._input_shape[-1],
            activation=self._activation,
            activity_regularizer=self._activity_regularizer,
            spectral_norm=self._spectral_norm,
            kernel_regularizer=self._kernel_regularizer,
            reconstruction_activation=self._reconstruction_activation
        )
        self._decoder.build((None, self._latent_dim))

        super().build(self._input_shape)

    @tf.function
    def train_step(self, data):
        """
        Overrides `train_step` to implement custom loss handling.
        """

        x, y = data
        with tf.GradientTape() as tape:

            z_mean, z_log_var, z = self._encoder.__call__(x)
            reconstruction = self._decoder.__call__(z)

            reconstruction_loss = self.compiled_loss(
                    tf.reshape(x, (-1, self._input_shape[-1])), reconstruction
            )

            if self._discrepancy_loss == "mmd":
                true_samples = tf.random.normal(tf.stack(
                        [self._input_shape[0], self._latent_dim]
                ))
                discrepancy_loss = self._mmd_loss(z, true_samples)
            elif self._discrepancy_loss == "kld":
                kld = -0.5 * (
                        1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
                )
                discrepancy_loss =  tf.reduce_mean(tf.reduce_sum(kld))
            else:
                raise ValueError("Only KLD and MMD currently supported as "
                                 "difference-in-distribution losses.")

            total_loss = reconstruction_loss + discrepancy_loss

        gradients = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))

        self._loss_tracker.update_state(total_loss)
        self._reconstruction_loss_tracker.update_state(reconstruction_loss)
        self._discrepancy_loss_tracker.update_state(discrepancy_loss)

        return {m.name: m.result() for m in self._tracked_metrics}

    @property
    def metrics(self):
        """
        Return tracked metrics here to facilitate `reset_states()`
        """
        return self._tracked_metrics

    def encode(self, inputs):
        """
        Return sampled latent values.
        """
        _, _, latent = self._encoder.__call__(inputs)
        return latent

    def call(self, inputs, training=False):
        """
        Return reconstructed inputs.
        """
        _, _, latent = self._encoder.__call__(inputs)
        return self._decoder.__call__(latent)

    def get_config(self):
        config = super().get_config()
        config.update({
            "encoding_dims": self._encoding_dims,
            "latent_dim": self._latent_dim,
            "activation": self._activation,
            "reconstruction_activation": self._reconstruction_activation,
            "activity_regularizer": self._activity_regularizer,
            "kernel_regularizer": self._kernel_regularizer,
            "spectral_norm": self._spectral_norm,
            "discrepancy_loss": self._discrepancy_loss
        })
        return config

    @property
    def encoder(self):
        return self._encoder

    @property
    def decoder(self):
        return self._decoder