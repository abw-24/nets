
from tensorflow.keras import layers


class Block(layers.Layer):
    """
    Base block class. Creates a generic call method referencing
    self._block_layers, which is utilized by each child class to
     store computations.
    """

    def __init__(self):
        super(Block, self).__init__()
        self._block_layers = []

    def call(self, inputs):
        x = inputs
        for lyr in self._block_layers:
            x = lyr(x)
        return x


class DenseBlock(Block):
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

    def get_config(self):
        return {
            "dims": self._dims,
            "activation": self._activation
        }


class CNNBlock(Block):
    """
    2d CNN Block (series of convolution and pooling ops, plus flattening).
    Assumes reshaping and input formatting is done already, and expects
    the input dim to be dynamically assigned with build().
    """

    def __init__(self, dims, stride=3, activation="relu", pool=True):

        super(CNNBlock, self).__init__()

        self._stride = stride
        self._activation = activation
        self._pool = pool

        if isinstance(dims, int):
            self._dims = [dims]
        else:
            self._dims = dims

        for d in self._dims:
            self._block_layers.append(
                    layers.Conv2D(d, self._stride, padding='same', activation=self._activation)
            )
            if self._pool:
                self._block_layers.append(layers.MaxPooling2D())

        self._block_layers.append(layers.Flatten())

    def get_config(self):
        return {
            "dims": self._dims,
            "stride": self._stride,
            "activation": self._activation,
            "pool": self._pool
        }


class Softmax(Block):
    """
    Output layer (for completeness).
    """

    def __init__(self, dims):
        super(Softmax, self).__init__()
        self._dims = dims
        self._block_layers = [layers.Dense(self._dims)]

    def get_config(self):
        return {
            "dims": self._dims,
        }