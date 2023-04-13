
import unittest
import tensorflow as tf

from nets.layers.sequence import MultiHeadSelfAttention as MHSA
from nets.tests.utils import try_except_assertion_decorator


#TODO: add tests with pooling and concat options
class TestMultiHeadSelfAttention(unittest.TestCase):

    def setUp(self):
        """
        Create fresh default params for each test.
        """
        self._input_shape = (16, 10, 256)
        self._num_heads = 2
        self._key_dim = 128
        self._masking = True

        self._default_config = {
            "num_heads": self._num_heads,
            "key_dim": self._key_dim,
            "masking": self._masking
        }

    @try_except_assertion_decorator
    def test_build_basic_with_constructor_defaults(self):
        _ = MHSA()

    @try_except_assertion_decorator
    def test_build_basic(self):
        _ = MHSA(
                num_heads=self._num_heads,
                key_dim=self._key_dim,
                masking=self._masking
        )

    @try_except_assertion_decorator
    def test_build_from_config(self):
        _ = MHSA.from_config(self._default_config)

    def test_build_get_config(self):
        block = MHSA(**self._default_config)
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
        block = MHSA(**self._default_config)
        output = block(inputs)
        assert output.shape == self._input_shape, \
            "MHA input shape: {}\nMHA output shape: {}".format(
                    self._input_shape, output.shape
            )