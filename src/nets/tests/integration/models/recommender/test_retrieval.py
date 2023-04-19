
import tensorflow as tf
import os
from unittest import TestCase as TC

from nets.models.recommender.retrieval import TwoTowerRetrieval
from nets.models.recommender.embedding import DeepHashEmbedding
from nets.models.recommender.embedding import HashEmbedding

from nets.tests.integration.models.base import ModelIntegrationABC, \
    RecommenderIntegrationTrait
from nets.tests.utils import *


class TestTwoTowerRetrieval(RecommenderIntegrationTrait, ModelIntegrationABC, TC):
    """
    """

    temp = os.path.join(os.getcwd(), "twotower-tmp-model")

    def _generate_default_compiled_model(self):
        """
        Instantiate and return a retrieval with the default params.
        """
        query_model = HashEmbedding(embedding_dim=self._embedding_dim)
        candidate_model = HashEmbedding(embedding_dim=self._embedding_dim)

        model = TwoTowerRetrieval(
                query_model=query_model,
                candidate_model=candidate_model,
                query_id=self._query_id,
                candidate_id=self._candidate_id
        )
        model.compile(
            optimizer=obj_from_config(tf.keras.optimizers, self._optimizer)
        )
        return model

    def _generate_deep_compiled_model(self):
        """
        Instantiate and return a deep retrieval model.
        """
        query_model = DeepHashEmbedding(embedding_dim=self._embedding_dim)
        candidate_model = DeepHashEmbedding(embedding_dim=self._embedding_dim)

        model = TwoTowerRetrieval(
                query_model=query_model,
                candidate_model=candidate_model,
                query_id=self._query_id,
                candidate_id=self._candidate_id
        )
        model.compile(
            optimizer=obj_from_config(tf.keras.optimizers, self._optimizer)
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
        model = self._generate_deep_compiled_model()
        model.fit(
                self._train,
                epochs=self._epochs,
                callbacks=[TrainSanityAssertionCallback()]
        )
