
import tensorflow as tf
import tensorflow_recommenders as tfrs
import os
from unittest import TestCase as TC

from nets.models.recommender.retrieval import TwoTowerRetrieval
from nets.models.recommender.embedding import DeepHashEmbedding
from nets.layers.recommender import HashEmbedding
from nets.utils import get_obj

from nets.tests.utils import try_except_assertion_decorator, \
    TrainSanityAssertionCallback
from nets.tests.integration.models.base import ModelIntegrationABC, \
    RecommenderIntegrationMixin


class TestTwoTowerRetrieval(RecommenderIntegrationMixin, ModelIntegrationABC, TC):
    """
    Fine tuning tester. For simplicity, here we simply create
    """

    temp = os.path.join(os.getcwd(), "twotower-tmp-model")

    def _generate_default_compiled_model(self):
        """
        Instantiate and return a retrieval with the default params.
        """
        user_model = HashEmbedding(embedding_dim=self._embedding_dim)
        item_model = HashEmbedding(embedding_dim=self._embedding_dim)

        model = TwoTowerRetrieval(
                user_model=user_model,
                item_model=item_model,
                user_features=self._user_features,
                item_features=self._item_features
        )
        model.compile(
            optimizer=get_obj(tf.keras.optimizers, self._optimizer)
        )
        return model

    def _generate_deep_compiled_model(self):
        """
        Instantiate and return a deep retrieval model.
        """
        user_model = DeepHashEmbedding(embedding_dim=self._embedding_dim)
        item_model = DeepHashEmbedding(embedding_dim=self._embedding_dim)

        model = TwoTowerRetrieval(
                user_model=user_model,
                item_model=item_model,
                user_features=self._user_features,
                item_features=self._item_features
        )
        model.compile(
            optimizer=get_obj(tf.keras.optimizers, self._optimizer)
        )
        return model

    @try_except_assertion_decorator
    def test_build_deep_embedding(self):
        """
        Generate a retrieval model with deep hash embeddings.
        """
        _ = self._generate_deep_compiled_model()

    def test_fit_deep_embedding(self):
        """
        Test that training "works" (by the definition of TrainSanityCallback)
        for a retrieval model with deep embeddings. Assertion is done directly
        in TrainSanityCallback.
        """
        model = self._generate_default_compiled_model()
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