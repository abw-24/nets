
"""
Factories for common blocks of built-in and/or library defined layers.
For library defined model factories (corresponding to nets model classes),
see nets.models.factory
"""


import tensorflow as tf

from nets.layers.conv import ResidualLayer, MultiPathResidualLayer


class ConvBlockFactory(object):
    
    @classmethod
    def apply(cls, config):
        """
        2d CNN Block (series of convolution and pooling ops). Assumes reshaping
         and input formatting is done prior. Note: if you wish to specify a non-square
         kernel, specify as a tuple.
        
        Configuration keys and values:
        :key filters: List of integers specifying the number of feature filters in
            each layer. The length implicitly specifies the number of
            convolution+pooling layers in the block (list)
        :key kernel: Kernel size for each filter (int | tuple)
        :key stride: Kernel stride (int)
        :key activation: Activation function (str)
        :key pool: Flag to add pooling layer after each convolution (bool)
        :key batch_norm: Flag to add batch normalization after each convolution (bool)
        
        
        :param config: Configuration (dict)
        :return: tf.keras.Sequential model
        """

        filters = config.get("filters")
        kernel = config.get("kernel", 3)
        stride = config.get("stride", (1,1))
        padding = config.get("padding", "same")
        activation = config.get("activation", "relu")
        pool = config.get("pool", True)
        batch_norm = config.get("batch_norm", False)

        assert isinstance(filters, list), "Value for key `filters` expected to be of type list."

        depth = len(filters)

        if isinstance(kernel, int) or (isinstance(kernel, tuple) and isinstance(kernel[0], int)):
            kernel = [kernel]*depth
        else:
            kernel = kernel
        if isinstance(stride, int) or (isinstance(stride, tuple) and isinstance(stride[0], int)):
            stride = [stride]*depth
        else:
            stride = stride
        if isinstance(padding, str) or (isinstance(padding, tuple) and isinstance(padding[0], int)):
            padding = [padding]*depth
        else:
            padding = padding

        layers = []
        for f, k, s, p in zip(filters, kernel, stride, padding):
            layers.append(tf.keras.layers.Conv2D(
                    f, k, strides=s, padding=p, activation=activation
            ))
            if batch_norm:
                layers.append(tf.keras.layers.BatchNormalization())
            if pool:
                layers.append(tf.keras.layers.MaxPooling2D())

        return tf.keras.Sequential(layers=layers)


class ResidualBlockFactory(object):
    """

    """

    @classmethod
    def apply(cls, config):
        """

        :param config: Configuration (dict)
        :return: tf.keras.Sequential model
        """

        block_depth = config.get("block_depth")
        block_filters = config.get("filters")
        res_activation = config.get("activation")

        if isinstance(block_filters, int):
            block_filters = [block_filters] * block_depth
        else:
            if isinstance(block_filters[0], int):
                block_filters = [block_filters] * block_depth

        layers = [ResidualLayer(b, activation=res_activation) for b in block_filters]

        return tf.keras.Sequential(layers=layers)
            

class MultiPathResidualBlockFactory(object):
    """

    """

    @classmethod
    def apply(cls, config):
        """

        :param config: Configuration (dict)
        :return: tf.keras.Sequential model
        """

        block_depth = config.get("block_depth")
        n_paths = config.get("n_paths")
        block_filters = config.get("filters")
        res_activation = config.get("activation")

        if isinstance(n_paths, int):
            n_paths = [n_paths]*block_depth

        if isinstance(block_filters, int):
            block_filters = [block_filters] * block_depth
        else:
            if isinstance(block_filters[0], int):
                block_filters = [block_filters] * block_depth

        layers = [
            MultiPathResidualLayer(p, filters=f, activation=res_activation)
            for p, f in zip(n_paths, block_filters)
        ]

        return tf.keras.Sequential(layers=layers)