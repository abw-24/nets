
import tensorflow as tf


@tf.keras.utils.register_keras_serializable("nets")
class DenseBlock(tf.keras.layers.Layer):

    def __init__(self, hidden_dims, activation="relu", kernel_regularizer=None,
                 activity_regularizer=None, **kwargs):

        super(DenseBlock, self).__init__(**kwargs)

        if isinstance(hidden_dims, int):
            self._hidden = [hidden_dims]
        else:
            self._hidden = hidden_dims
        if isinstance(activation, str):
            self._activation = [activation]*len(self._hidden)
        else:
            self._activation = activation

        self._kernel_regularizer = kernel_regularizer
        self._activity_regularizer = activity_regularizer

        self._block_layers = [
            tf.keras.layers.Dense(
                    units=d,
                    activation=a,
                    kernel_regularizer=self._kernel_regularizer,
                    activity_regularizer=self._activity_regularizer
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
