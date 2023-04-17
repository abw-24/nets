
from unittest import TestCase as TC
import tensorflow as tf
import os

from nets.models.vae import GaussianDenseVAE

from nets.tests.integration.models.base import ModelIntegrationABC, \
    DenseIntegrationTrait
from nets.tests.utils import obj_from_config, try_except_assertion_decorator, \
    TrainSanityAssertionCallback


class TestVAE(DenseIntegrationTrait, ModelIntegrationABC, TC):

    temp = os.path.join(os.getcwd(), "vae-tmp-model")

    def setUp(self):
        """
        Create fresh default params for each test.
        """
        self._input_shape = (32, 784)
        self._encoding_dims = [8]
        self._activation = "relu"
        self._latent_dim = 4
        self._reconstruction_activation = "sigmoid"
        self._optimizer = {"Adam": {"learning_rate": 0.001}}
        self._loss = {"MeanSquaredError": {}}
        self._epochs = 1

    def _generate_default_compiled_model(self):
        """
        Instantiate and return a model with the default params and compiled
        with the default loss and optimizer.
        """
        model = GaussianDenseVAE(
            encoding_dims=self._encoding_dims,
            latent_dim=self._latent_dim,
            activation=self._activation,
            reconstruction_activation=self._reconstruction_activation
        )
        model.build(input_shape=self._input_shape)
        model.compile(
            optimizer=obj_from_config(tf.keras.optimizers, self._optimizer),
            loss=obj_from_config(tf.keras.losses, self._loss)
        )
        return model

    @try_except_assertion_decorator
    def test_build_no_build(self):
        """
        Test that model creation works when specifying the input shape in the
         model constructor (triggering a call of `build` on construction).
        """
        model = GaussianDenseVAE(
                input_shape=self._input_shape,
                encoding_dims=self._encoding_dims,
                latent_dim=self._latent_dim,
                activation=self._activation,
                reconstruction_activation=self._reconstruction_activation
        )
        model.compile(
            optimizer=obj_from_config(tf.keras.optimizers, self._optimizer),
            loss=obj_from_config(tf.keras.losses, self._loss)
        )

    def test_fit_complex(self):
        """
        Test that training "works" (by the definition of TrainSanityCallback)
        for a more complex model with a different compiled loss and optimizer.
        Assertion is done directly in TrainSanityCallback.
        :return:
        """
        optimizer = {"RMSprop": {"learning_rate": 0.001}}
        loss = {"MeanAbsoluteError": {}}
        activity_regularizer = {"L2": {}}
        encoding_dims = [64, 32, 16]
        spectral_norm = True

        model = GaussianDenseVAE(
                encoding_dims=encoding_dims,
                activation=self._activation,
                latent_dim=self._latent_dim,
                reconstruction_activation=self._reconstruction_activation,
                activity_regularizer=obj_from_config(tf.keras.regularizers, activity_regularizer),
                spectral_norm=spectral_norm,
            )
        model.build(input_shape=self._input_shape)
        model.compile(
            optimizer=obj_from_config(tf.keras.optimizers, optimizer),
            loss=obj_from_config(tf.keras.losses, loss)
        )
        model.fit(
                self._train,
                epochs=self._epochs,
                callbacks=[TrainSanityAssertionCallback()]
        )

    @try_except_assertion_decorator
    def test_save_and_load(self):
        """
        Test that saving and loading works. Here, we test the encoder submodel
        as opposed to the full model for two reasons:

         (1) Keeping only the encoder portion for later inference is by far the
          most frequent use-case.

         (2) Saving the entire model will not work as is -- the model's `call`
         method must return only the reconstructed inputs to have
         high-level keras API methods like `predict` work as expected,
         but the custom `train_step` needs both the reconstructions and the
         distribution parameters for the latent space to compute the composite
         loss metric for training. As a result, `self.__call__` cannot be used
         to compute the full forward pass in `train_step`, and saving fails.
         This could potentially be worked around by re-implementing `call`
         and `predict_on_batch` if it is of interest.
        """

        model = self._generate_default_compiled_model()
        model.fit(
                self._train,
                epochs=self._epochs
        )
        model.encoder.save(self.temp)
        _ = tf.keras.models.load_model(self.temp)

    @try_except_assertion_decorator
    def test_save_and_load_decoder(self):
        """
        Test that saving and loading works. Here we test the `decoder` submodel
        to complement the `encoder` submodel test above.
        """

        model = self._generate_default_compiled_model()
        model.fit(
                self._train,
                epochs=self._epochs
        )
        model.decoder.save(self.temp)
        _ = tf.keras.models.load_model(self.temp)