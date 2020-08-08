
from tensorflow.keras import layers as klayers
import tensorflow as tf


class SequentialBlock(klayers.Layer):
    """
    Base block class. Creates a generic call method referencing
    self._block_layers, which is utilized by each child class to
     store computations. Layers that use the `training` argument
     to change computations will of course need to overwrite
     the call method.
    """

    def __init__(self, **kwargs):
        super(SequentialBlock, self).__init__(**kwargs)
        self._block_layers = []

    def call(self, inputs, training=False):
        x = inputs
        for lyr in self._block_layers:
            x = lyr(x, training=training)
        return x


class DenseBlock(SequentialBlock):
    """
    Block of densely connected layers.
    """

    def __init__(self, dims, activation, **kwargs):

        super(DenseBlock, self).__init__(**kwargs)

        if isinstance(dims, int):
            self._dims = [dims]
        else:
            self._dims = dims
        if isinstance(activation, str):
            self._activation = [activation]*len(self._dims)
        else:
            self._activation = activation

        self._block_layers = [
            klayers.Dense(u, a) for u, a in zip(self._dims, self._activation)
            ]

    def get_config(self):
        config = super(DenseBlock, self).get_config()
        config.update({
            "dims": self._dims,
            "activation": self._activation
        })
        return config


class ConvBlock(SequentialBlock):

    def __init__(
            self,
            filters,
            kernel=3,
            stride=(1,1),
            padding="same",
            activation="relu",
            pool=True,
            batch_norm=False,
            **kwargs
    ):
        """
        2d CNN Block (series of convolution and pooling ops). Assumes reshaping and
        input formatting is done already, and expects the input dim to be
        dynamically assigned with build(). Note: if you wish to specify a non-square
        kernel (shame on you), specify as a tuple.

        :param filters: List of integers specifying the number of feature filters in
            each layer. The length implicitly specifies the number of
            convolution+pooling layers in the block.
        :param kernel: Kernel size for each filter (int or tuple)
        :param stride: Kernel stride (int)
        :param activation: Activation function (str)
        :param pool: Flag to add pooling layer after each convolution
        :param batch_norm: Flag to add batch normalization after each convolution
        :return:
        """

        super(ConvBlock, self).__init__(**kwargs)

        self._pool_flag = pool
        self._activation = activation
        self._batch_norm_flag = batch_norm

        if isinstance(filters, int):
            self._filters = [filters]
        else:
            self._filters = filters
        if isinstance(kernel, int) or isinstance(kernel, tuple):
            self._kernel = [kernel]*len(filters)
        else:
            self._kernel = kernel
        if isinstance(stride, int) or isinstance(stride, tuple):
            self._stride = [stride]*len(filters)
        else:
            self._stride = stride
        if isinstance(padding, str) or isinstance(padding, tuple):
            self._padding = [padding]*len(filters)
        else:
            self._padding = padding

        for f, k, s, p in zip(self._filters, self._kernel, self._stride, self._padding):
            self._block_layers.append(
                    klayers.Conv2D(f, k, strides=s, padding=p, activation=self._activation)
            )
            if self._batch_norm_flag:
                self._block_layers.append(klayers.BatchNormalization())
            if self._pool_flag:
                self._block_layers.append(klayers.MaxPooling2D())

    def get_config(self):
        config = super(ConvBlock, self).get_config()
        config.update({
            "filters": self._filters,
            "kernel": self._kernel,
            "stride": self._stride,
            "padding": self._padding,
            "activation": self._activation,
            "pool": self._pool_flag,
            "batch_norm": self._batch_norm_flag
        })
        return config


class ResidualLayer(klayers.Layer):

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
        self._conv_layer = ConvBlock(
                filters,
                kernel=3,
                stride=(1,1),
                activation=self._conv_activation,
                padding="same",
                pool=False,
                batch_norm=False
        )
        self._add_op = klayers.Add()
        self._activation_op = klayers.Activation(self._conv_activation)

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


class MultiPathResidualLayer(klayers.Layer):

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
        self._conv_paths = []

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
            self._conv_paths.append(ConvBlock(
                    filters=[int(f/pow(2, i)) for f in self._filters[:-1]] + [self._filters[-1]],
                    kernel=3,
                    stride=(1,1),
                    activation=self._conv_activation,
                    padding="same",
                    pool=False,
                    batch_norm=True
            ))

        self._add_op = klayers.Add()
        self._activation_op = klayers.Activation(self._conv_activation)

    def call(self, inputs, training=False):
        x = inputs
        path_outputs = []
        for path in self._conv_paths:
            path_out = path(x, training=training)
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


class ResidualBlock(SequentialBlock):
    """

    """

    def __init__(self, block_depth, filters, activation="relu", **kwargs):

        super(ResidualBlock, self).__init__(**kwargs)

        self._block_depth = block_depth
        self._block_filters = filters
        self._res_activation = activation

        self._block_layers = []

        if isinstance(self._block_filters, int):
            self._block_filters = [self._block_filters] * self._block_depth
        else:
            if isinstance(self._block_filters[0], int):
                self._block_filters = [self._block_filters] * self._block_depth

        for i in range(self._block_depth):
            self._block_layers.append(
                ResidualLayer(self._block_filters[i], self._res_activation)
            )

    def get_config(self):
        config = super(ResidualBlock, self).get_config()
        config.update({
            "block_depth": self._block_depth,
            "filters": self._block_filters,
            "activation": self._res_activation,
        })
        return config


class MultiPathResidualBlock(SequentialBlock):
    """

    """

    def __init__(self, block_depth, n_paths, filters, activation="relu", **kwargs):

        super(MultiPathResidualBlock, self).__init__(**kwargs)

        self._block_depth = block_depth
        self._n_paths = n_paths
        self._block_filters = filters
        self._res_activation = activation

        if isinstance(n_paths, int):
            self._n_paths = [self._n_paths]*self._block_depth

        if isinstance(self._block_filters, int):
            self._block_filters = [self._block_filters] * self._block_depth
        else:
            if isinstance(self._block_filters[0], int):
                self._block_filters = [self._block_filters] * self._block_depth

        for i in range(self._block_depth):
            self._block_layers.append(
                MultiPathResidualLayer(
                        self._n_paths[i],
                        filters=self._block_filters[i],
                        activation=self._res_activation
                )
            )

    def get_config(self):
        config = super(MultiPathResidualBlock, self).get_config()
        config.update({
            "block_depth": self._block_depth,
            "n_paths": self._n_paths,
            "filters": self._block_filters,
            "activation": self._res_activation,
        })
        return config


class VAESampling(klayers.Layer):
    """
    Samples from distribution defined by the latent layer values to
    generate values from which to decode.
    """

    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.keras.backend.random_normal(
                shape=(tf.shape(z_mean)[0],
                       tf.shape(z_mean)[1])
        )
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def get_config(self):
        return super(VAESampling, self).get_config()


class DenseEncoder(klayers.Layer):

    def __init__(self, mapping_dims, latent_dim, activation="relu", **kwargs):

        super(DenseEncoder, self).__init__(**kwargs)

        self._mapping_dims = mapping_dims
        self._latent_dim = latent_dim
        self._activation = activation

        self._encode_block = DenseBlock(
                self._mapping_dims,
                activation=self._activation
        )
        self._latent_mean = klayers.Dense(self._latent_dim)
        self._latent_log_var = klayers.Dense(self._latent_dim)
        self._sampling = VAESampling()

    def call(self, inputs):
        x = self._encode_block(inputs)
        z_mean = self._latent_mean(x)
        z_log_var = self._latent_log_var(x)
        z = self._sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z

    def get_config(self):
        config = super(DenseEncoder, self).get_config()
        config.update({
            "mapping_dims": self._mapping_dims,
            "latent_dim": self._latent_dim,
            "activation": self._activation
        })
        return config


class DenseDecoder(klayers.Layer):

    def __init__(self, inverse_mapping_dims, input_dim, activation="relu", **kwargs):

        super(DenseDecoder, self).__init__(**kwargs)

        self._inverse_mapping_dims = inverse_mapping_dims
        self._input_dim = input_dim
        self._activation = activation

        self._decode_block = DenseBlock(
                self._inverse_mapping_dims,
                activation=self._activation
        )
        self._output = klayers.Dense(self._input_dim)

    def call(self, inputs):
        x = self._decode_block(inputs)
        return self._output(x)

    def get_config(self):
        config = super(DenseDecoder, self).get_config()
        config.update({
            "inverse_mapping_dims": self._inverse_mapping_dims,
            "input_dim": self._input_dim,
            "activation": self._activation
        })
        return config
