import unittest

import numpy as np
import os
import shutil
import tensorflow as tf
from nets.models.vae import GaussianDenseVAE
from nets.utils import get_obj

from nets.tests.utils import *


class TestVAE(unittest.TestCase):

    temp = os.path.join(os.getcwd(), "vae-encoder-tmp-model")

    @classmethod
    def setUpClass(cls):
        """
        Load training data from keras once for all tests.
        """
        # Load mnist data, flatten, and normalize to 0-1
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = x_train.reshape((-1, 784))
        x_train = x_train / 255.0

        # Create a batch feed from the train tensors
        cls._train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)) \
            .shuffle(10000) \
            .batch(32)

        # Keep the test xs as well
        cls._x_test = x_test.reshape(-1, 784)

    @classmethod
    def tearDownClass(cls):
        """
        Delete training data, saved model.
        """
        del cls._train_ds
        del cls._x_test

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

    def tearDown(self):
        """
        If we saved something (a model), delete it.
        :return:
        """
        if os.path.exists(self.temp):
            shutil.rmtree(self.temp)

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
            optimizer=get_obj(tf.keras.optimizers, self._optimizer),
            loss=get_obj(tf.keras.losses, self._loss)
        )
        return model

    @try_except_assertion_decorator
    def test_build_basic(self):
        """
        Test that default model creation works.
        """
        _ = self._generate_default_compiled_model()

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
            optimizer=get_obj(tf.keras.optimizers, self._optimizer),
            loss=get_obj(tf.keras.losses, self._loss)
        )

    def test_fit_basic(self):
        """
        Test that training "works" (by the definition of TrainSanityCallback)
        for the default model. Assertion is done directly in
        TrainSanityCallback.
        """
        model = self._generate_default_compiled_model()
        model.fit(
                self._train_ds,
                epochs=self._epochs,
                callbacks=[TrainSanityAssertionCallback()]
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
                activity_regularizer=get_obj(tf.keras.regularizers, activity_regularizer),
                spectral_norm=spectral_norm,
            )
        model.build(input_shape=self._input_shape)
        model.compile(
            optimizer=get_obj(tf.keras.optimizers, optimizer),
            loss=get_obj(tf.keras.losses, loss)
        )
        model.fit(
                self._train_ds,
                epochs=self._epochs,
                callbacks=[TrainSanityAssertionCallback()]
        )

    def test_predict(self):
        """
        Test that prediction works and returns the right type.
        """

        model = self._generate_default_compiled_model()
        model.fit(
                self._train_ds,
                epochs=self._epochs
        )
        predictions = model.predict(self._x_test)

        assert isinstance(predictions, np.ndarray)\
               or isinstance(predictions, tf.Tensor)

    @try_except_assertion_decorator
    def test_save_and_load_encoder(self):
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
                self._train_ds,
                epochs=self._epochs
        )
        model.encoder.save(self.temp)
        _ = tf.keras.models.load_model(self.temp)

    @try_except_assertion_decorator
    def test_save_and_load_decoder(self):
        """
        Test that saving and loading works. Here, we test the decoder submodel
        to complement the `encoder` submodel test above.
        """

        model = self._generate_default_compiled_model()
        model.fit(
                self._train_ds,
                epochs=self._epochs
        )
        model.decoder.save(self.temp)
        _ = tf.keras.models.load_model(self.temp)