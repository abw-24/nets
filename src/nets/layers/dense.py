
import tensorflow as tf
import tensorflow_addons as tfa


@tf.keras.utils.register_keras_serializable("nets")
class DenseBlock(tf.keras.layers.Layer):

    def __init__(self, hidden_dims, activation="relu", kernel_regularizer=None,
                 activity_regularizer=None, spectral_norm=False, **kwargs):

        super(DenseBlock, self).__init__(**kwargs)

        if isinstance(hidden_dims, int):
            self._hidden_dims = [hidden_dims]
        else:
            self._hidden_dims = hidden_dims
        if isinstance(activation, str):
            self._activation = [activation]*len(self._hidden_dims)
        else:
            self._activation = activation

        self._kernel_regularizer = kernel_regularizer
        self._activity_regularizer = activity_regularizer
        self._spectral_norm = spectral_norm

        self._block_layers = []
        for d, a in zip(self._hidden_dims, self._activation):
            layer = tf.keras.layers.Dense(
                    units=d,
                    activation=a,
                    kernel_regularizer=self._kernel_regularizer,
                    activity_regularizer=self._activity_regularizer
            )

            if self._spectral_norm:
                layer = tfa.layers.SpectralNormalization(layer)

            self._block_layers.append(layer)

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
            "hidden_dims": self._hidden_dims,
            "activation": self._activation,
            "activity_regularizer": self._activity_regularizer,
            "kernel_regularizer": self._kernel_regularizer,
            "spectral_norm": self._spectral_norm
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
