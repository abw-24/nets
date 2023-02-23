
"""
MLP.
"""

import tensorflow as tf

from nets.blocks.dense import DenseSequentialBlockFactory
from nets.models.base import BaseModel
from nets.utils import tf_shape_to_list


@tf.keras.utils.register_keras_serializable("nets")
class MLP(BaseModel):

    def __init__(self, hidden_dims, output_dim, input_dim=None, activation="relu",
                 output_activation="softmax", kernel_regularizer=None,
                 activity_regularizer=None, name="MLP", **kwargs):

        super().__init__(name=name,  **kwargs)

        self._hidden_dims = hidden_dims
        self._output_dim = output_dim
        self._input_dim = input_dim
        self._activation = activation
        self._output_activation = output_activation
        self._kernel_regularizer = kernel_regularizer
        self._activity_regularizer = activity_regularizer

        if self._input_dim is not None:
            self.build(input_shape=self._input_dim)

    def build(self, input_shape):
        """
        Build model graph.

        :param input_shape: Shape of input tensor (tuple)
        """
        # Factory creates a tf.keras.Sequential model
        self._forward = DenseSequentialBlockFactory.apply(
            hidden=self._hidden_dims,
            activation=self._activation,
            input_shape=tf_shape_to_list(input_shape),
            kernel_regularizer=self._kernel_regularizer,
            activity_regularizer=self._activity_regularizer
        )
        # Add the output layer on the end
        self._forward.add(
                tf.keras.layers.Dense(
                        units=self._output_dim,
                        activation=self._output_activation
                )
        )

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
            y_hat = self._forward(x, training=True)
            loss = self.compiled_loss(y, y_hat)
        # Compute gradients
        gradients = tape.gradient(loss, self._forward.trainable_variables)
        # Apply gradients
        self.optimizer.apply_gradients(
                zip(gradients, self._forward.trainable_variables)
        )
        # Update the compiled metrics
        self.compiled_metrics.update_state(y, y_hat)
        return {m.name: m.result() for m in self.metrics}

    def call(self, inputs):
        return self._forward(inputs)

    def get_config(self):
        config = super(MLP, self).get_config()
        config.update({
            "hidden_dims": self._hidden_dims,
            "output_dim": self._output_dim,
            "input_dim": self._self._input_dim,
            "activation": self._activation,
            "output_activation": self._output_activation,
            "kernel_regularizer": self._kernel_regularizer,
            "activity_regularizer": self._activity_regularizer
        })
        return config
