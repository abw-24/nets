
import unittest

import numpy as np
import os
import shutil
import tensorflow as tf
import tensorflow_recommenders as tfrs
import tensorflow_datasets as tfds
from nets.models.recommender.twotower import SimpleEmbeddingTwoTower
from nets.utils import get_obj

from nets.tests.utils import *


class TestSimpleEmbeddingTwoTower(unittest.TestCase):

    temp = os.path.join(os.getcwd(), "twotower-tmp-model")

    @classmethod
    def setUpClass(cls):
        """
        Load training data once for all tests.
        """
        tf.random.set_seed(1)
        # Load data
        ratings = tfds.load("movielens/100k-ratings", split="train")
        movies = tfds.load("movielens/100k-movies", split="train")

        ratings = ratings\
            .map(lambda x: {
                "movie_title": x["movie_title"],
                "user_id": x["user_id"],
            })


        shuffled_ratings = ratings.shuffle(10000, seed=1, reshuffle_each_iteration=False)

        cls._train = shuffled_ratings.take(10000).batch(100).cache()

        cls._movies = movies\
            .map(lambda x: x["movie_title"])\
            .batch(100)

        cls._user_ids = ratings.batch(100)\
            .map(lambda x: x["user_id"])

        cls._items = np.unique(np.concatenate(list(cls._movies)))
        cls._users = np.unique(np.concatenate(list(cls._user_ids)))

    @classmethod
    def tearDownClass(cls):
        """
        Delete training data, saved model.
        """
        del cls._train
        del cls._movies
        del cls._user_ids
        del cls._users
        del cls._items
        if os.path.exists(cls.temp):
            shutil.rmtree(cls.temp)

    def setUp(self):
        """
        Create fresh default params for each test.
        """
        self._input_shape = (32, 784)
        self._embedding_dim = 32
        self._user_features = "user_id"
        self._item_features = "movie_title"
        self._activation = "relu"
        self._optimizer = {"Adam": {"learning_rate": 0.001}}
        self._task = tfrs.tasks.Retrieval()
        self._epochs = 1

    def _generate_default_compiled_model(self):
        """
        Instantiate and return a model with the default params and compiled
        with the default loss and optimizer defined in setUp
        """
        model = SimpleEmbeddingTwoTower(
                task=self._task,
                embedding_dim=self._embedding_dim,
                users=self._users,
                items=self._items,
                user_features=self._user_features,
                item_features=self._item_features
        )
        model.compile(
            optimizer=get_obj(tf.keras.optimizers, self._optimizer)
        )
        return model

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

    def test_fit_ranking(self):
        """
        Test that training "works" (by the definition of TrainSanityCallback)
        for the default model. Assertion is done directly in
        TrainSanityCallback.
        """
        ranking_task = tfrs.tasks.Ranking()

        model = SimpleEmbeddingTwoTower(
                task=ranking_task,
                embedding_dim=self._embedding_dim,
                users=self._users,
                items=self._items,
                user_features=self._user_features,
                item_features=self._item_features
        )
        model.compile(
            optimizer=get_obj(tf.keras.optimizers, self._optimizer)
        )
        model.fit(
                self._train,
                epochs=self._epochs,
                callbacks=[TrainSanityAssertionCallback()]
        )

    @try_except_assertion_decorator
    def test_predict(self):
        """
        Test that prediction works.
        """
        model = self._generate_default_compiled_model()
        model.fit(
                self._train,
                epochs=self._epochs
        )
        index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
        index.index_from_dataset(
                tf.data.Dataset.zip((
                    self._movies, self._movies.map(model.item_model)
                ))
        )

        _, titles = index(tf.constant(["1"]))

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
        index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
        index.index_from_dataset(
                tf.data.Dataset.zip((
                    self._movies, self._movies.map(model.item_model)
                ))
        )

        tf.saved_model.save(index, self.temp)
        _ = tf.saved_model.load(self.temp)