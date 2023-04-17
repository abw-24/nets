
from unittest import TestCase as TC
import tensorflow as tf
import numpy as np
import os

from nets.models.mlp import MLP
from nets.utils import get_obj

from nets.tests.integration.models.base import \
    DenseIntegrationTrait, ModelIntegrationABC
from nets.tests.utils import try_except_assertion_decorator, \
    TrainSanityAssertionCallback


class TestMLP(DenseIntegrationTrait, ModelIntegrationABC, TC):

    temp = os.path.join(os.getcwd(), "mlp-tmp-model")

    def setUp(self):
        """
        Create fresh default params for each test.
        """
        self._input_shape = (32, 784)
        self._hidden_dims = [32, 16]
        self._activation = "relu"
        self._output_dim = 10
        self._output_activation = "softmax"
        self._optimizer = {"Adam": {"learning_rate": 0.001}}
        self._loss = {"MeanSquaredError": {}}
        self._epochs = 1

    def _generate_default_compiled_model(self):
        """
        Instantiate and return a model with the default params and compiled
        with the default loss and optimizer defined in setUp
        """
        model = MLP(
            hidden_dims=self._hidden_dims,
            output_dim=self._output_dim,
            activation=self._activation,
            output_activation=self._output_activation
        )
        model.build(input_shape=self._input_shape)
        model.compile(
            optimizer=get_obj(tf.keras.optimizers, self._optimizer),
            loss=get_obj(tf.keras.losses, self._loss)
        )
        return model

    @try_except_assertion_decorator
    def test_build_no_build(self):
        """
        Test that model creation works when specifying input shape in the model
        parameters as opposed to later invoking .build() manually
        """
        model = MLP(
            input_shape=self._input_shape,
            hidden_dims=self._hidden_dims,
            activation=self._activation,
            output_dim=self._output_dim,
            output_activation=self._output_activation
        )
        model.compile(
            optimizer=get_obj(tf.keras.optimizers, self._optimizer),
            loss=get_obj(tf.keras.losses, self._loss)
        )

    def test_fit_complex(self):
        """
        Test that training "works" (by the definition of TrainSanityCallback)
        for a more complex model with a different compiled loss and optimizer.
        Assertion is done directly in TrainSanityCallback.
        """
        optimizer = {"RMSprop": {"learning_rate": 0.001}}
        loss = {"MeanAbsoluteError": {}}
        activity_regularizer =  {"L2": {}}
        hidden_dims = [64, 32, 16]
        spectral_norm = True

        model = MLP(
                hidden_dims=hidden_dims,
                activation=self._activation,
                output_dim=self._output_dim,
                output_activation=self._output_activation,
                activity_regularizer=get_obj(
                        tf.keras.regularizers, activity_regularizer
                ),
                spectral_norm=spectral_norm
            )
        model.build(input_shape=self._input_shape)
        model.compile(
            optimizer=get_obj(tf.keras.optimizers, optimizer),
            loss=get_obj(tf.keras.losses, loss)
        )
        model.fit(
                self._train,
                epochs=self._epochs,
                callbacks=[TrainSanityAssertionCallback()]
        )

