
import unittest

from nets.layers.dense import DenseBlock
from nets.utils import get_obj
from nets.tests.utils import *


class TestDense(unittest.TestCase):

    def setUp(self):
        """
        Create fresh default params for each test.
        """
        self._input_shape = (32, 784)
        self._hidden_dims = [32, 16]
        self._activation = "relu"

        self._default_config = {
            "hidden_dims": self._hidden_dims,
            "activation": self._activation
        }

    @try_except_assertion_decorator
    def test_build_basic_with_constructor_defaults(self):
        _ = DenseBlock(hidden_dims=self._hidden_dims)

    @try_except_assertion_decorator
    def test_build_basic(self):
        _ = DenseBlock(hidden_dims=self._hidden_dims, activation=self._activation)

    @try_except_assertion_decorator
    def test_build_complex(self):
        activity_regularizer = get_obj(tf.keras.regularizers, {"L2": {}})
        kernel_regularizer = get_obj(tf.keras.regularizers, {"L2": {}})
        _ = DenseBlock(
            hidden_dims=self._hidden_dims,
            activation=self._activation,
            activity_regularizer=activity_regularizer,
            kernel_regularizer=kernel_regularizer
        )

    @try_except_assertion_decorator
    def test_build_from_config(self):
        _ = DenseBlock.from_config(self._default_config)

    def test_build_get_config(self):
        block = DenseBlock(**self._default_config)
        c = block.get_config()
        assert isinstance(c, dict)
        assert all([k in c for k, v in self._default_config.items()]), \
            """
            Missing a passed model param in the model config definition.
            Passed configuration: {}
            Returned configuration: {}
            """.format(self._default_config, c)