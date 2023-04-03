
import tensorflow as tf
import tensorflow_recommenders as tfrs
import os
from unittest import TestCase as TC

from nets.models.recommender.retrieval import TwoTowerRetrieval, \
    ListwiseTwoTowerRetrieval
from nets.models.recommender.embedding import DeepHashEmbedding
from nets.layers.recommender import HashEmbedding
from nets.utils import get_obj

from nets.tests.utils import try_except_assertion_decorator, \
    TrainSanityAssertionCallback
from nets.tests.integration.models.base import ModelIntegrationABC, \
    RecommenderIntegrationMixin, ListwiseRecommenderIntegrationMixin


class TestTwoTowerRetrieval(RecommenderIntegrationMixin, ModelIntegrationABC, TC):
    """
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
                user_id=self._user_id,
                item_id=self._item_id
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
                user_id=self._user_id,
                item_id=self._item_id
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


class TestListwiseTwoTowerRetrieval(ListwiseRecommenderIntegrationMixin, TestTwoTowerRetrieval):
    """
    Need to overwrite setUpClass to reformat training data.
    Otherwise identical to regular two tower testing.
    """

    temp = os.path.join(os.getcwd(), "twotower-tmp-model")

    def _generate_default_compiled_model(self):
        """
        Instantiate and return a retrieval with the default params.
        """
        user_model = HashEmbedding(embedding_dim=self._embedding_dim)
        item_model = HashEmbedding(embedding_dim=self._embedding_dim)

        model = ListwiseTwoTowerRetrieval(
                user_model=user_model,
                item_model=item_model,
                user_id=self._user_id,
                item_id=self._item_id
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

        model = ListwiseTwoTowerRetrieval(
                user_model=user_model,
                item_model=item_model,
                user_id=self._user_id,
                item_id=self._item_id
        )
        model.compile(
            optimizer=get_obj(tf.keras.optimizers, self._optimizer)
        )
        return model