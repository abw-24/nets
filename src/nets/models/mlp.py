
"""
MLP.
"""

import tensorflow as tf

from nets.layers.dense import DenseBlock
from nets.models.base import BaseTFKerasModel


@tf.keras.utils.register_keras_serializable("nets")
class MLP(BaseTFKerasModel):

    def __init__(self, hidden_dims, output_dim, input_shape=None,
                 activation="relu", output_activation="softmax",
                 kernel_regularizer=None, activity_regularizer=None,
                 spectral_norm=False, name="MLP", **kwargs):

        super().__init__(name=name,  **kwargs)

        self._hidden_dims = hidden_dims
        self._output_dim = output_dim
        self._input_shape = input_shape
        self._activation = activation
        self._output_activation = output_activation
        self._kernel_regularizer = kernel_regularizer
        self._activity_regularizer = activity_regularizer
        self._spectral_norm = spectral_norm

        self._dense_block = DenseBlock(
            hidden_dims=self._hidden_dims,
            activation=self._activation,
            kernel_regularizer=self._kernel_regularizer,
            activity_regularizer=self._activity_regularizer,
            spectral_norm=self._spectral_norm
        )

        self._output_layer = tf.keras.layers.Dense(
                self._output_dim,
                activation=self._output_activation
        )

        self._input_layer = None
        if self._input_shape is not None:
            self.build(self._input_shape)

    def build(self, input_shape):

        self._input_layer = tf.keras.layers.InputLayer(
                input_shape=(input_shape[-1],)
        )
        super().build(input_shape)

    def call(self, inputs, training=True):
        return self._output_layer.__call__(
                self._dense_block.__call__(
                        self._input_layer.__call__(inputs)
                )
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_dims": self._hidden_dims,
            "output_dim": self._output_dim,
            "input_shape": self._input_shape,
            "activation": self._activation,
            "output_activation": self._output_activation,
            "kernel_regularizer": self._kernel_regularizer,
            "activity_regularizer": self._activity_regularizer,
            "spectral_norm": self._spectral_norm
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
