import unittest

import tensorflow as tf
from nets.models.factory import GaussianDenseVAEFactory, MLPFactory
from nets.utils import get_obj

from nets.tests.utils import try_except_assertion_decorator


class TestFactory(unittest.TestCase):

    def setUp(self):
        """
        Create fresh default configs for each test.
        """
        self._input_shape = (32, 784)
        self._generic_config = {
            "activation": "relu",
            "optimizer": {"Adam": {"learning_rate": 0.001}},
            "epochs": 1
        }
        self._default_mlp_config = self._generic_config.copy()
        self._default_mlp_config.update({
            "hidden_dims": [32, 16],
            "output_dim": 10,
            "output_activation": "softmax",
            "loss": {"SparseCategoricalCrossentropy": {}},
        })
        self._default_vae_config = self._generic_config.copy()
        self._default_vae_config.update({
            "encoding_dims": [32, 16],
            "latent_dim": 10,
            "reconstruction_activation": "relu",
            "loss": {"MeanSquaredError": {}},
        })

    @try_except_assertion_decorator
    def test_mlp_factory_basic(self):
        mlp = MLPFactory.apply(self._default_mlp_config)
        mlp.build(self._input_shape)

    @try_except_assertion_decorator
    def test_mlp_factory_complex(self):
        self._default_mlp_config.update({
            "input_shape": self._input_shape,
            "activity_regularizer": get_obj(tf.keras.regularizers, {"L2": {}}),
            "kernel_regularizer": get_obj(tf.keras.regularizers, {"L2": {}})
        })
        _ = MLPFactory.apply(self._default_mlp_config)

    @try_except_assertion_decorator
    def test_vae_factory_basic(self):
        vae = GaussianDenseVAEFactory.apply(self._default_vae_config)
        vae.build(self._input_shape)

    @try_except_assertion_decorator
    def test_vae_factory_complex(self):
        self._default_vae_config.update({
                "input_shape": self._input_shape,
                "activity_regularizer": get_obj(tf.keras.regularizers, {"L2": {}}),
                "kernel_regularizer": get_obj(tf.keras.regularizers, {"L2": {}})
        })
        _ = GaussianDenseVAEFactory.apply(self._default_vae_config)
