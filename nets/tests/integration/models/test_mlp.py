
import tensorflow as tf
import unittest
import numpy as np

from nets.models.mlp import MLP
from nets.utils import get_obj
from nets.tests.utils import TrainSanityCallback


class TestMLP(unittest.TestCase):

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
        Delete training data.
        :return:
        """
        del cls._train_ds
        del cls._x_test

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
        :return:
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

    def test_build_basic(self):
        """
        Test that basic model creation works with the default model
        :return:
        """
        try:
            model = self._generate_default_compiled_model()
        except Exception as e:
            success = False
            msg = e
        else:
            success = True
            msg = "Success!"

        assert success, msg

    def test_build_no_build(self):
        """
        Test that model creation works when specifying input shape in the model
        parameters as opposed to later invoking .build() manually
        :return:
        """
        try:
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
        except Exception as e:
            success = False
            msg = e
        else:
            success = True
            msg = "Success!"

        assert success, msg

    def test_train_basic(self):
        """
        Test that training "works" (by the definition of TrainSanityCallback)
        for the default model. Assertion is done directly in
        TrainSanityCallback.
        :return:
        """

        model = self._generate_default_compiled_model()
        model.fit(
                self._train_ds,
                epochs=self._epochs,
                callbacks=[TrainSanityCallback()]
        )

    def test_train_complex(self):
        """
        Test that training "works" (by the definition of TrainSanityCallback)
        for a more complex model with a different compiled loss and optimizer.
        Assertion is done directly in TrainSanityCallback.
        :return:
        """
        optimizer = {"RMSprop": {"learning_rate": 0.001}}
        loss = {"MeanAbsoluteError": {}}
        activity_regularizer =  {"L2": {}}
        hidden_dims = [64, 32, 16]

        model = MLP(
                hidden_dims=hidden_dims,
                activation=self._activation,
                output_dim=self._output_dim,
                output_activation=self._output_activation,
                activity_regularizer=get_obj(tf.keras.regularizers, activity_regularizer)
            )
        model.build(input_shape=self._input_shape)
        model.compile(
            optimizer=get_obj(tf.keras.optimizers, optimizer),
            loss=get_obj(tf.keras.losses, loss)
        )
        model.fit(
                self._train_ds,
                epochs=self._epochs,
                callbacks=[TrainSanityCallback()]
        )

    def test_predict(self):
        """
        Test that prediction works and returns the right type.
        :return:
        """
        model = self._generate_default_compiled_model()
        model.fit(
                self._train_ds,
                epochs=self._epochs,
                callbacks=[TrainSanityCallback()]
        )
        predictions = model.predict(self._x_test)

        assert isinstance(predictions, np.ndarray)\
               or isinstance(predictions, tf.Tensor)
