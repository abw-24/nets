
import tensorflow as tf

from nets.layers.sampling import GaussianSampling
from nets.blocks.dense import DenseSequentialBlockFactory


@tf.keras.utils.register_keras_serializable("nets")
class DenseGaussianVariationalEncoder(tf.keras.layers.Layer):

    def __init__(self, encoding_dims, latent_dim, input_dim=None,
                 activation="relu", activity_regularizer=None,
                 sparse_flag=False, name="VAE", **kwargs):

        super().__init__(name=name, **kwargs)

        self._encoding_dims = encoding_dims
        self._latent_dim = latent_dim
        self._input_dim = input_dim
        self._activation = activation
        self._a_regularizer = activity_regularizer
        self._sparse_flag = sparse_flag

        if self._input_dim is not None:
            self.build(input_shape=self._input_dim)

    def build(self, input_shape):
        """
        Build model graph.

        :param input_shape: Shape of input tensor (tuple)
        """

        self._encoding_block = DenseSequentialBlockFactory.apply(
                hidden=self._encoding_dims,
                activation=self._activation,
                input_shape=input_shape,
                activity_regularizer=self._activity_regularizer
        )
        self._latent_mean = tf.keras.layers.Dense(self._latent_dim)
        self._latent_log_var = tf.keras.layers.Dense(self._latent_dim)
        self._sampling = GaussianSampling()

    def call(self, inputs):
        """

        :param inputs:
        :return:
        """
        x = self._encoding_block(inputs)
        z_mean = self._latent_mean(x)
        z_log_var = self._latent_log_var(x)
        z = self._sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z

    def get_config(self):
        config = super(DenseGaussianVariationalEncoder, self).get_config()
        config.update({
            "encoding_dims": self._encoding_dims,
            "latent_dim": self._latent_dim,
            "input_dim": self._input_dim,
            "activation": self._activation,
            "activity_regularizer": self._activity_regularizer,
            "sparse_flag": self._sparse_flag
        })
        return config