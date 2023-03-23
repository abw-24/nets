
import tensorflow as tf
import tensorflow_addons as tfa


@tf.keras.utils.register_keras_serializable
class ConvBlock(tf.keras.layers.Layer):

    def __init__(self, filters, kernel=3, stride=(1,1), padding="same",
                 activation="relu", pool=True, batch_norm=False,
                 spectral_norm=True, **kwargs):
        """
        2d CNN Block (series of convolution and pooling ops). Assumes reshaping
         and input formatting is done already, and expects the input shape to
         be assigned with build(). Note: if you wish to specify a non-square
        kernel, specify as a tuple.

        :param filters: List of integers specifying the number of feature
            filters in each layer. The length implicitly specifies the
            number of convolution+pooling layers in the block (list)
        :param kernel: Kernel size for each filter (int | tuple)
        :param stride: Kernel stride (int)
        :param activation: Activation function (str)
        :param pool: Flag to add pooling layer after each convolution (bool)
        :param batch_norm: Flag to add batch normalization after each convolution (bool)
        :param spectral_norm: Flag to apply spectral normalization to conv layer weights (bool)
        :return:
        """

        super(ConvBlock, self).__init__(**kwargs)

        self._pool_flag = pool
        self._activation = activation
        self._batch_norm = batch_norm
        self._spectral_norm = spectral_norm

        if isinstance(filters, int):
            self._filters = [filters]
        else:
            self._filters = filters

        depth = len(filters)

        if isinstance(kernel, int) or isinstance(kernel, tuple):
            self._kernel = [kernel]*depth
        else:
            self._kernel = kernel
        if isinstance(stride, int) or isinstance(stride, tuple):
            self._stride = [stride]*depth
        else:
            self._stride = stride
        if isinstance(padding, str) or isinstance(padding, tuple):
            self._padding = [padding]*depth
        else:
            self._padding = padding

        self._block_layers = []
        for f, k, s, p in zip(self._filters, self._kernel, self._stride, self._padding):

            conv = tf.keras.layers.Conv2D(
                    f, k, strides=s, padding=p, activation=self._activation
            )

            if self._spectral_norm:
                self._block_layers.append(tfa.layers.SpectralNormalization(conv))
            else:
                self._block_layers.append(conv)

            if self._batch_norm:
                self._block_layers.append(tf.keras.layers.BatchNormalization())
            if self._pool_flag:
                self._block_layers.append(tf.keras.layers.MaxPooling2D())

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
        config = super(ConvBlock, self).get_config()
        config.update({
            "filters": self._filters,
            "kernel": self._kernel,
            "stride": self._stride,
            "padding": self._padding,
            "activation": self._activation,
            "pool": self._pool_flag,
            "batch_norm": self._batch_norm,
            "spectral_norm": self._spectral_norm
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


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
        self._conv_layer = ConvBlock(filters=filters)
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

    @classmethod
    def from_config(cls, config):
        return cls(**config)


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
            self._conv_paths[str(i)] = ConvBlock(
                filters=[int(f/pow(2, i)) for f in self._filters[:-1]] + [self._filters[-1]],
                activation=self._conv_activation
            )

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

    @classmethod
    def from_config(cls, config):
        return cls(**config)


