
from tensorflow.keras import layers as klayers


class SequentialBlock(klayers.Layer):
    """
    Base block class. Creates a generic call method referencing
    self._block_layers, which is utilized by each child class to
     store computations. Layers that use the `training` argument
     to change computations will of course need to overwrite
     the call method.
    """

    def __init__(self):
        super(SequentialBlock, self).__init__()
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

    def __init__(self, dims, activation):

        super(DenseBlock, self).__init__()

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


class ConvBlock(SequentialBlock):

    def __init__(
            self,
            filters,
            kernel=3,
            stride=(1,1),
            padding="same",
            activation="relu",
            pool=True,
            batch_norm=False
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

        super(ConvBlock, self).__init__()

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
                self._block_layers.append(klayers.BatchNormalization(axis=3))
            if self._pool_flag:
                self._block_layers.append(klayers.MaxPooling2D())


class ResidualLayer(klayers.Layer):

    def __init__(self, filters, activation="relu"):
        """
        Single 2D convolutional block (of configurable depth) plus a residual connection
        at the end. Assumes "same" padding, so residual connection dims will match.

        :param filters: List of integers specifying the number of feature filters in
            each layer. The length implicitly specifies the number of
            convolution layers in the block.
        :param activation: Activation function (str)
        """

        super(ResidualLayer).__init__()

        self._activation = activation
        self._conv_layer = ConvBlock(
                filters,
                kernel=3,
                stride=(1,1),
                activation=self._activation,
                padding="same",
                pool=False,
                batch_norm=True
        )
        self._residual_layer = klayers.Add()
        self._activation_layer = klayers.Activation(self._activation)

    def call(self, inputs, training=False):
        x = inputs
        conv_out = self._conv_layer(x, training=training)
        res_out = self._residual_layer([x, conv_out])
        output = self._activation_layer(res_out)
        return output


class MultiPathResidualLayer(klayers.Layer):

    def __init__(self, n_paths, filters, activation="relu"):
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

        super(MultiPathResidualLayer).__init__()

        self._activation = activation
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

        for i in range(n_paths):
            self._conv_paths.append(ConvBlock(
                    filters=[int(f/pow(2, i)) for f in self._filters[:-1]] + [self._filters[-1]],
                    kernel=3,
                    stride=(1,1),
                    activation=self._activation,
                    padding="same",
                    pool=False,
                    batch_norm=True
            ))

        self._add_op = klayers.Add()
        self._activation_layer = klayers.Activation(self._activation)

    def call(self, inputs, training=False):
        x = inputs
        path_outputs = []
        for path in self._conv_paths:
            path_out = path(x, training=training)
            path_outputs.append(path_out)
        merged = self._add_op(path_outputs)
        output = self._add_op([x, merged])
        output = self._activation_layer(output)
        return output


class ResidualBlock(SequentialBlock):
    """

    """

    def __init__(self, block_depth, filters, activation="relu"):

        super(ResidualBlock, self).__init__()

        self._block_depth = block_depth
        self._filters = filters
        self._activation = activation

        self._block_layers = []

        if isinstance(self._filters, int):
            self._filters = [self._filters]*self._block_depth
        else:
            if isinstance(self._filters[0], int):
                self._filters = [self._filters]*self._block_depth

        for i in range(self._block_depth):
            self._block_layers.append(
                ResidualLayer(self._filters[i], self._activation)
            )


class MultiPathResidualBlock(SequentialBlock):
    """

    """

    def __init__(self, block_depth, n_paths, filters, activation="relu"):

        super(MultiPathResidualBlock, self).__init__()

        self._block_depth = block_depth
        self._n_paths = n_paths
        self._filters = filters
        self._activation = activation

        self._block_layers = []

        if isinstance(n_paths, int):
            self._n_paths = [self._n_paths]*self._block_depth

        if isinstance(self._filters, int):
            self._filters = [self._filters]*self._block_depth
        else:
            if isinstance(self._filters[0], int):
                self._filters = [self._filters]*self._block_depth

        for i in range(self._block_depth):
            self._block_layers.append(
                MultiPathResidualLayer(
                        self._n_paths[i],
                        filters=self._filters[i],
                        activation=self._activation
                )
            )
