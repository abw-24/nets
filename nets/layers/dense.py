
import tensorflow as tf

from nets.layers.sampling import GaussianSampling


@tf.keras.utils.register_keras_serializable("nets")
class DenseBlock(tf.keras.layers.Layer):

    def __init__(self, hidden, activation="relu", kernel_regularizer=None,
                 activity_regularizer=None, **kwargs):

        super(DenseBlock, self).__init__(**kwargs)

        if isinstance(hidden, int):
            self._hidden = [hidden]
        else:
            self._hidden = hidden
        if isinstance(activation, str):
            self._activation = [activation]*len(self._hidden)
        else:
            self._activation = activation

        self._block_layers = [
            tf.keras.layers.Dense(
                    units=d,
                    activation=a,
                    kernel_regularizer=kernel_regularizer,
                    activity_regularizer=activity_regularizer
            )
            for d, a in zip(self._hidden, self._activation)
        ]

    def call(self, inputs, training=False):
        """

        :param inputs:
        :return:
        """
        outputs = inputs
        for layer in self._block_layers:
            outputs = layer(outputs)
        return outputs

    def get_config(self):
        config = super(DenseBlock, self).get_config()
        config.update({
            "hidden": self._hidden,
            "activation": self._activation,
            "activity_regularizer": self._activity_regularizer,
            "kernel_regularizer": self._kernel_regularizer
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable("nets")
class DenseGaussianVariationalEncoder(tf.keras.layers.Layer):

    def __init__(self, encoding_dims, latent_dim, activation="relu",
                 activity_regularizer=None, sparse_flag=False, name="VAE",
                 **kwargs):

        super().__init__(name=name, **kwargs)

        self._encoding_dims = encoding_dims
        self._latent_dim = latent_dim
        self._activation = activation
        self._a_regularizer = activity_regularizer
        self._sparse_flag = sparse_flag

        self._encoding_block = DenseBlock(
                hidden=self._encoding_dims,
                activation=self._activation,
                activity_regularizer=self._activity_regularizer
        )
        self._latent_mean = tf.keras.layers.Dense(self._latent_dim)
        self._latent_log_var = tf.keras.layers.Dense(self._latent_dim)
        self._sampling = GaussianSampling()

    def call(self, inputs, training=False):
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
            "input_dim": self._input_shape,
            "activation": self._activation,
            "activity_regularizer": self._activity_regularizer,
            "sparse_flag": self._sparse_flag
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)