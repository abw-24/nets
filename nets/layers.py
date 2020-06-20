
from tensorflow.keras import layers


class SequentialBlock(layers.Layer):
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

    def call(self, inputs, training=None):
        x = inputs
        for lyr in self._block_layers:
            x = lyr(x)
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
            layers.Dense(u, a) for u, a in zip(self._dims, self._activation)
            ]


class CNNBlock(SequentialBlock):

    def __init__(self, filters, kernel=3, stride=(1,1), activation="relu", pool=True):
        """
        2d CNN Block (series of convolution and pooling ops, plus flattening).
        Assumes reshaping and input formatting is done already, and expects
        the input dim to be dynamically assigned with build(). Note: if
        you wish to specify a non-square kernel (shame on you), specify as a
        tuple.

        :param filters: List of integers specifying the number of feature filters in
            each layer. The length implicitly specifies the number of
            convolution+pooling layers in the block.
        :param kernel: Kernel size for each filter (int or tuple)
        :param stride: Kernel stride (int)
        :param activation: Activation function (str)
        :param pool: Flag to add pooling layer after each convolution
        """

        super(CNNBlock, self).__init__()

        self._pool = pool
        self._activation = activation

        # the length of the filter array specifies the number of convolution layers
        if isinstance(filters, int):
            self._filters = [filters]
        else:
            self._filters = filters
        # fill out a list of the appropriate depth if not provided
        if isinstance(kernel, int) or isinstance(kernel, tuple):
            self._kernel = [kernel]*len(filters)
        else:
            self._kernel = kernel
        # fill out a list of the appropriate depth if not provided
        if isinstance(stride, int):
            self._stride = [stride]*len(filters)
        else:
            self._stride = stride

        for f, k, s in zip(self._filters, self._kernel, self._stride):
            self._block_layers.append(
                    layers.Conv2D(f, k, strides=s, padding="same", activation=self._activation)
            )
            if self._pool:
                self._block_layers.append(layers.MaxPooling2D())

        self._block_layers.append(layers.Flatten())