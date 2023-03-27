
import tensorflow as tf
import tensorflow_recommenders as tfrs
import os
from unittest import TestCase as TC, skip

from nets.models.recommender.twotower import SimpleEmbeddingTwoTower, \
    FineTuningTwoTower
from nets.layers.recommender import StringEmbedding
from nets.utils import get_obj

from nets.tests.utils import try_except_assertion_decorator, \
    TrainSanityAssertionCallback
from nets.tests.integration.models.base import ModelIntegrationABC, \
    RecommenderIntegrationMixin


class TestSimpleEmbeddingTwoTower(RecommenderIntegrationMixin, ModelIntegrationABC, TC):

    temp = os.path.join(os.getcwd(), "simpletwotower-tmp-model")

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


class TestFineTuningTwoTower(RecommenderIntegrationMixin, ModelIntegrationABC, TC):
    """
    Fine tuning tester. For simplicity, here we simply create
    """

    temp = os.path.join(os.getcwd(), "finetuningtwotower-tmp-model")

    def _generate_default_compiled_model(self):
        """
        Instantiate and return a model with the default params and compiled
        with the default loss and optimizer defined in setUp.
        """
        user_model = StringEmbedding(
                vocab=self._users, embedding_dim=self._embedding_dim
        )
        item_model = StringEmbedding(
                vocab=self._items, embedding_dim=self._embedding_dim
        )
        model = FineTuningTwoTower(
                task=self._task,
                user_model=user_model,
                item_model=item_model,
                user_features=self._user_features,
                item_features=self._item_features
        )
        model.compile(
            optimizer=get_obj(tf.keras.optimizers, self._optimizer)
        )
        return model

    def test_fit_ranking(self):
        """
        Test that training "works" (by the definition of TrainSanityCallback)
        for the default model. Assertion is done directly in
        TrainSanityCallback.
        """
        ranking_task = tfrs.tasks.Ranking()

        user_model = StringEmbedding(
                vocab=self._users, embedding_dim=self._embedding_dim
        )
        item_model = StringEmbedding(
                vocab=self._items, embedding_dim=self._embedding_dim
        )
        model = FineTuningTwoTower(
                task=ranking_task,
                user_model=user_model,
                item_model=item_model,
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
