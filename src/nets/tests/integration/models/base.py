
from abc import abstractmethod, ABC
import os
import shutil
import numpy as np

import tensorflow as tf
import tensorflow_recommenders as tfrs
import tensorflow_datasets as tfds

from nets.tests.utils import try_except_assertion_decorator, \
    TrainSanityAssertionCallback


class ModelIntegrationABC(ABC):

    """
    Model integration test ABC. Defines the following --

        - `setUpClass` should load and store data for model train testing
        - `tearDownClass` should clean up any data stored and remaining
        model artifacts on disk
        - `setUp` should create default params for model creation and compilation
        -  `_generate_default_compiled_model` should create, compile, and
        return a default model (defined by default params from `setUp`)
        - `test_build` should test building a default model
        - `test_fit` should test fitting a default model
        - `test_predict` should test the `predict` method of the model
        - `test_save_and_load` should test the `save` and `load` functionality
         of either the top-level model, or a submodel (whichever is appropriate
          for the given model)
    """

    temp = None
    _train = None
    _epochs = 1

    @classmethod
    @abstractmethod
    def setUpClass(cls):
        """
        Create fresh default params for each test.
        """
        raise NotImplementedError("Abstract")

    @classmethod
    @abstractmethod
    def tearDownClass(cls):
        """
        Create fresh default params for each test.
        """
        raise NotImplementedError("Abstract")

    @abstractmethod
    def setUp(self):
        """
        Create fresh default params for each test.
        """
        raise NotImplementedError("Abstract")

    @abstractmethod
    def _generate_default_compiled_model(self):
        """
        Create, compile, and return the appropriate model from the default
         params set up in `setUp`
        :return:
        """
        raise NotImplementedError("Abstract")

    @abstractmethod
    def test_build(self):
        raise NotImplementedError("Abstract")

    @abstractmethod
    def test_fit(self):
        raise NotImplementedError("Abstract")

    @abstractmethod
    def test_predict(self):
        raise NotImplementedError("Abstract")

    @abstractmethod
    def test_save_and_load(self):
        raise NotImplementedError("Abstract")


class ModelIntegrationMixin(object):
    """
    Common concrete methods for all integration mixin flavors.
        - `tearDown` deletes any artifacts saved at `temp` (may need to be
        overwritten by certain children)
        - `test_build` is a generic test for creating the default model
        - `test_fit` is a generic test for fitting the default model
    """

    def tearDown(self):
        """
        If we saved something (a model), delete it.
        """
        if os.path.exists(self.temp):
            shutil.rmtree(self.temp)

    @try_except_assertion_decorator
    def test_build(self):
        """
        Test that basic model creation works with the default model
        """
        _ = self._generate_default_compiled_model()

    def test_fit(self):
        """
        Test that training "works" (by the definition of TrainSanityCallback)
        for the default model. Assertion is done directly in
        TrainSanityCallback.
        """
        model = self._generate_default_compiled_model()
        model.fit(
                self._train,
                epochs=self._epochs,
                callbacks=[TrainSanityAssertionCallback()]
        )


class RecommenderIntegrationMixin(ModelIntegrationMixin):

    """
    Base mixin for recommender model testing. Implements several of the base
    ABC's abstract methods:
        - `setUpClass` reads and preps movielens data
        - `tearDownClass` deletes movielens data and any artifacts at temp path
        - `setUp` creates fresh default params for each test method (may need
        to be ovewritten for certain models)
        - `test_predict` uses the default model to test if prediction works
        as expected
        - `test_save_and_load` uses the default model to test if saving/loading
        works as expected
    """

    @classmethod
    def setUpClass(cls):
        """
        Load training data once for all tests.
        """
        tf.random.set_seed(1)

        ratings = tfds.load("movielens/100k-ratings", split="train")
        movies = tfds.load("movielens/100k-movies", split="train")

        ratings = ratings\
            .map(lambda x: {
                "movie_title": x["movie_title"],
                "user_id": x["user_id"],
                "user_rating": x["user_rating"]
            })

        shuffled_ratings = ratings.shuffle(10000, seed=1, reshuffle_each_iteration=False)

        cls._train = shuffled_ratings.take(10000).batch(100).cache()

        cls._movies = movies\
            .map(lambda x: x["movie_title"])\
            .batch(100)

        cls._user_ids = ratings\
            .map(lambda x: x["user_id"])\
            .batch(100)

    @classmethod
    def tearDownClass(cls):
        """
        Delete training data, any remaining artifacts.
        """
        del cls._train
        del cls._movies
        del cls._user_ids

        if os.path.exists(cls.temp):
            shutil.rmtree(cls.temp)

    def setUp(self):
        """
        Create fresh default params for each test.
        """
        self._embedding_dim = 32
        self._query_id = "user_id"
        self._candidate_id = "movie_title"
        self._rank_target = "user_rating"
        self._activation = "relu"
        self._optimizer = {"Adam": {"learning_rate": 0.001}}
        self._task = tfrs.tasks.Retrieval()
        self._epochs = 1

    @try_except_assertion_decorator
    def test_predict(self):
        """
        Test that prediction works. Assumes no context model.
        """
        model = self._generate_default_compiled_model()
        model.fit(
                self._train,
                epochs=self._epochs
        )
        index = tfrs.layers.factorized_top_k.BruteForce(k=10)
        index.index_from_dataset(
                tf.data.Dataset.zip((
                    self._movies.map(lambda x: model.query_model((x, None))),
                    self._movies.map(lambda x: model.candidate_model((x, None)))
                ))
        )

        _, titles = index(model.query_model((tf.constant(["1"]), None)))

    @try_except_assertion_decorator
    def test_save_and_load(self):
        """
        Test that saving and loading works.
        """

        model = self._generate_default_compiled_model()
        model.fit(
                self._train,
                epochs=self._epochs
        )
        index = tfrs.layers.factorized_top_k.BruteForce(k=10)
        index.index_from_dataset(
                tf.data.Dataset.zip((
                    self._movies.map(lambda x: model.query_model((x, None))),
                    self._movies.map(lambda x: model.candidate_model((x, None)))
                ))
        )

        tf.saved_model.save(index, self.temp)
        _ = tf.saved_model.load(self.temp)


class ListwiseRecommenderIntegrationMixin(RecommenderIntegrationMixin):
    """
    Listwise flavor of recommender integration Mixin. Uses tfrs helpers
    to reformat ratings data for listwise loss. Currently does not work
    as expected...
    """

    #TODO: `tfrs.examples.movielens.sample_listwise` needs replacing?
    @classmethod
    def setUpClass(cls):
        tf.random.set_seed(1)

        ratings = tfds.load("movielens/100k-ratings", split="train")
        movies = tfds.load("movielens/100k-movies", split="train")

        ratings = ratings\
            .map(lambda x: {
                "movie_title": x["movie_title"],
                "user_id": x["user_id"],
                "user_rating": x["user_rating"]
            })\
            .take(10000)

        cls._train = tfrs.examples.movielens.sample_listwise(
            ratings,
            num_list_per_user=5,
            num_examples_per_list=5,
            seed=1
        )

        cls._movies = movies\
            .map(lambda x: x["movie_title"])\
            .batch(100)

        cls._user_ids = ratings\
            .map(lambda x: x["user_id"])\
            .batch(100)


class DenseIntegrationMixin(ModelIntegrationMixin):

    """
    Mixin for dense model testing. Implements several of the base ABCs
    abstract methods:
        - `setUpClass` reads and preps mnist data
        - `tearDownClass` deletes mnist movielens data
    """

    @classmethod
    def setUpClass(cls):
        """
        Load training data from keras once for all tests.
        """
        # Load mnist data, flatten, and normalize to 0-1
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = x_train.reshape((-1, 784))
        x_train = x_train / 255.0
        x_test = x_test.reshape((-1, 784))
        x_test = x_test / 255.0

        # Create a batch feed from the train tensors
        cls._train = tf.data.Dataset.from_tensor_slices((x_train, y_train)) \
            .shuffle(10000) \
            .batch(32)

        # Keep the test xs as well
        cls._x_test = x_test

    @classmethod
    def tearDownClass(cls):
        """
        Delete training data, any remaining artifacts.
        """
        del cls._train
        del cls._x_test

        if os.path.exists(cls.temp):
            shutil.rmtree(cls.temp)

    def test_predict(self):
        """
        Test that prediction works and returns the right type.
        """
        model = self._generate_default_compiled_model()
        model.fit(
                self._train,
                epochs=self._epochs,
                callbacks=[TrainSanityAssertionCallback()]
        )
        predictions = model.predict(self._x_test)

        assert isinstance(predictions, np.ndarray)\
               or isinstance(predictions, tf.Tensor)

    @try_except_assertion_decorator
    def test_save_and_load(self):
        """
        Test that saving and loading works.
        """

        model = self._generate_default_compiled_model()
        model.fit(
                self._train,
                epochs=self._epochs,
                callbacks=[TrainSanityAssertionCallback()]
        )
        model.save(self.temp)
        _ = tf.keras.models.load_model(self.temp)
