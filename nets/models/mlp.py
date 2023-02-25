
"""
MLP.
"""

import tensorflow as tf

from nets.layers.dense import DenseBlock
from nets.models.base import BaseModel


@tf.keras.utils.register_keras_serializable("nets")
class MLP(BaseModel):

    def __init__(self, hidden_dims, output_shape, input_dim=None, activation="relu",
                 output_activation="softmax", kernel_regularizer=None,
                 activity_regularizer=None, name="MLP", **kwargs):

        super().__init__(name=name,  **kwargs)

        self._hidden_dims = hidden_dims
        self._output_dim = output_shape
        self._input_shape = input_dim
        self._activation = activation
        self._output_activation = output_activation
        self._kernel_regularizer = kernel_regularizer
        self._activity_regularizer = activity_regularizer

        self._input_layer = None

        if self._input_shape is not None:
            self.build(self._input_shape)

        self._dense_block = DenseBlock(
            hidden=self._hidden_dims,
            activation=self._activation,
            kernel_regularizer=self._kernel_regularizer,
            activity_regularizer=self._activity_regularizer
        )
        self._output_layer = tf.keras.layers.Dense(
                self._output_dim,
                activation=self._output_activation
        )

    def build(self, input_shape):
        """

        :param input_shape:
        :return:
        """
        self._input_layer = tf.keras.layers.InputLayer(
                input_shape=(input_shape[-1],)
        )
        super().build(input_shape)

    def train_step(self, data):
        """
        Overrides parent method to train on a batch of data.

        :param data: (x, y) tuple, where x and y are batches of inputs and
            outputs (tuple)
        :return: Tracked metrics (dict)
        """

        x, y = data
        # Use the graph for the forward pass and compute the compiled loss
        with tf.GradientTape() as tape:
            y_hat = self._output_layer(self._dense_block(self._input_layer(x)))
            loss = self.compiled_loss(y, y_hat)
        # Compute gradients
        gradients = tape.gradient(loss, self._dense_block.trainable_variables)
        # Apply gradients
        self.optimizer.apply_gradients(
                zip(gradients, self._dense_block.trainable_variables)
        )
        # Update the compiled metrics
        self.compiled_metrics.update_state(y, y_hat)
        return {m.name: m.result() for m in self.metrics}

    def call(self, inputs):
        return self._output_layer(self._dense_block(self._input_layer(inputs)))

    def get_config(self):
        config = super(MLP, self).get_config()
        config.update({
            "hidden_dims": self._hidden_dims,
            "output_dim": self._output_dim,
            "input_dim": self._input_dim,
            "activation": self._activation,
            "output_activation": self._output_activation,
            "kernel_regularizer": self._kernel_regularizer,
            "activity_regularizer": self._activity_regularizer
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)