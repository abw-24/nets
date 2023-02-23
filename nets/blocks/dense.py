
"""
Factories for common blocks of built-in and/or library defined layers.
For library defined model factories (corresponding to nets model classes),
see nets.models.factory
"""

import tensorflow as tf


class DenseSequentialBlockFactory(object):

    @classmethod
    def apply(cls, hidden, activation, input_shape=None, kernel_regularizer=None,
              activity_regularizer=None):
        """
        Construct a dense feedforward network as a tf.keras.Sequential model.

        :param hidden: Hidden dims (list)
        :param activation: Activation (str | list)
        :param input_shape: Optional input dimension (int | None)
        :param kernel_regularizer: Optional kernal regularizer (str | None)
        :param activity_regularizer: Optional activity regularizer (str | None)
        :return: Keras Sequential model
        """
        depth = len(hidden)

        if isinstance(activation, str):
            activation = [activation]*depth

        assert depth == len(activation), \
            "Parameter `activation` expected to be the same length as " \
            "parameter `hidden`."

        layers = []

        if input_shape is not None:
            layers.append(tf.keras.layers.InputLayer(input_shape=input_shape))

        for h, a, in zip(*(hidden, activation)):
            layers.append(
                tf.keras.layers.Dense(
                        units=h,
                        activation=a,
                        activity_regularizer=kernel_regularizer,
                        kernel_regularizer=activity_regularizer
                )
            )

        return tf.keras.Sequential(layers)
