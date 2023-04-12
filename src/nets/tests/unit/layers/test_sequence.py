
import unittest
import tensorflow as tf

from nets.layers.sequence import MultiHeadMaskedSelfAttention as MHA
from nets.tests.utils import try_except_assertion_decorator


class TestMultiHeadMaskedSelfAttention(unittest.TestCase):

    def setUp(self):
        """
        Create fresh default params for each test.
        """
        self._input_shape = (16, 10, 256)
        self._num_heads = 2
        self._key_dim = 128

        self._default_config = {
            "num_heads": self._num_heads,
            "key_dim": self._key_dim
        }

    @try_except_assertion_decorator
    def test_build_basic_with_constructor_defaults(self):
        _ = MHA()

    @try_except_assertion_decorator
    def test_build_basic(self):
        _ = MHA(
                num_heads=self._num_heads,
                key_dim=self._key_dim
        )

    @try_except_assertion_decorator
    def test_build_from_config(self):
        _ = MHA.from_config(self._default_config)

    def test_build_get_config(self):
        block = MHA(**self._default_config)
        c = block.get_config()
        assert isinstance(c, dict)
        assert all([k in c for k, v in self._default_config.items()]), \
            """
            Missing a passed model param in the model config definition.
            Passed configuration: {}
            Returned configuration: {}
            """.format(self._default_config, c)

    def test_call(self):
        inputs = tf.random.normal(self._input_shape)
        block = MHA(**self._default_config)
        output = block(inputs)
        assert output.shape == self._input_shape, \
            "MHA input shape: {}\nMHA output shape: {}".format(
                    self._input_shape, output.shape
            )