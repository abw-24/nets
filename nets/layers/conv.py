
import tensorflow as tf

from nets.blocks.conv import ConvBlockFactory


@tf.keras.utils.register_keras_serializable("nets")
class ResidualLayer(tf.keras.layers.Layer):

    def __init__(self, filters, activation="relu", **kwargs):
        """
        Single 2D convolutional block (of configurable depth) plus a residual connection
        at the end. Assumes "same" padding, so residual connection dims will match.

        :param filters: List of integers specifying the number of feature filters in
            each layer. The length implicitly specifies the number of
            convolution layers in the block.
        :param activation: Activation function (str)
        """

        super(ResidualLayer, self).__init__(**kwargs)

        self._conv_activation = activation
        self._conv_layer = ConvBlockFactory.apply(config={"filters": filters})
        self._add_op = tf.keras.layers.Add()
        self._activation_op = tf.keras.layers.Activation(self._conv_activation)

    def call(self, inputs, training=False):
        x = inputs
        conv_out = self._conv_layer(x, training=training)
        res_out = self._add_op([x, conv_out])
        output = self._activation_op(res_out)
        return output

    def get_config(self):
        config = super(ResidualLayer, self).get_config()
        config.update({
            "filters": self._filters,
            "activation": self._activation,
        })
        return config


@tf.keras.utils.register_keras_serializable("nets")
class MultiPathResidualLayer(tf.keras.layers.Layer):

    def __init__(self, n_paths, filters, activation="relu", **kwargs):
        """
        Multiple 2D convolutional block paths (of configurable depth) merged
        plus a residual connection. Assumes "same" padding, so residual
        connection dims will match. The number of filters is altered (divided by a factor
        of 2) in secondary paths to encourage information flow.

        :param n_paths: Number of convolutional paths
        :param filters: List of integers specifying the number of feature filters in
            each conv layer (for each path). The length implicitly specifies the number of
            convolution layers in the block.
        :param activation: Activation function (str)
        """

        super(MultiPathResidualLayer, self).__init__(**kwargs)

        self._n_paths = n_paths
        self._conv_activation = activation
        self._conv_paths = {}

        if isinstance(filters, int):
            self._filters = [filters]
        else:
            self._filters = filters

        # change the number of filters in the secondary paths to
        # try to encourage different/more helpful representations.
        # in this case, we just divide each subsequent path by
        # an increasing power of 2. we make sure to use the same
        # number for the last conv op in each path so we can merge
        # them before the final residual connection

        for i in range(self._n_paths):
            self._conv_paths[str(i)] = ConvBlockFactory.apply({
                "filters": [int(f/pow(2, i)) for f in self._filters[:-1]] + [self._filters[-1]],
                "activation": self._conv_activation
            })

        self._add_op = tf.keras.layers.Add()
        self._activation_op = tf.keras.layers.Activation(self._conv_activation)

    def call(self, inputs, training=False):
        x = inputs
        path_outputs = []
        for i in range(self._n_paths):
            path_out = self._conv_paths[str(i)](x, training=training)
            path_outputs.append(path_out)
        merged = self._add_op(path_outputs)
        output = self._add_op([x, merged])
        output = self._activation_op(output)
        return output

    def get_config(self):
        config = super(MultiPathResidualLayer, self).get_config()
        config.update({
            "n_paths": self._n_paths,
            "filters": self._filters,
            "activation": self._activation,
        })
        return config


