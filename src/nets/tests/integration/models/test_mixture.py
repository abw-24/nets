
from unittest import TestCase as TC
import tensorflow as tf
import os

from nets.models.mixture import GatedMixtureABC
from nets.models.mlp import MLP

from nets.tests.integration.models.base import \
    DenseIntegrationTrait, ModelIntegrationABC
from nets.tests.utils import obj_from_config


# Test model using the mixture trait
@tf.keras.utils.register_keras_serializable("nets")
class SimpleMixture(GatedMixtureABC):

    def __init__(self, **kwargs):

        hidden_dims = [32]
        activation = "relu"
        output_dim = 10
        output_activation = "softmax"

        super().__init__(2, output_dim, **kwargs)

        # Simple mixture of MLPs
        model1 = MLP(
                hidden_dims=hidden_dims,
                output_dim=output_dim,
                activation=activation,
                output_activation=output_activation
        )
        model2 = MLP(
                hidden_dims=hidden_dims,
                output_dim=output_dim,
                activation=activation,
                output_activation=output_activation
        )
        self._experts = [model1, model2]

    def get_config(self):
        return {}


class TestGatedMixture(DenseIntegrationTrait, ModelIntegrationABC, TC):

    temp = os.path.join(os.getcwd(), "mixture-tmp-model")

    def setUp(self):
        """
        Create fresh default params for each test.
        """
        self._input_shape = (32, 784)
        self._optimizer = {"Adam": {"learning_rate": 0.001}}
        self._loss = {"MeanSquaredError": {}}
        self._epochs = 1

    def _generate_default_compiled_model(self):
        """
        Instantiate and return a model with the default params and compiled
        with the default loss and optimizer defined in setUp
        """

        mixture = SimpleMixture()
        mixture.build(input_shape=self._input_shape)
        mixture.compile(
                loss=obj_from_config(tf.keras.losses, self._loss),
                optimizer=obj_from_config(tf.keras.optimizers, self._optimizer)
        )

        return mixture
