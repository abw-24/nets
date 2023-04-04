

import tensorflow as tf
import os
from unittest import TestCase as TC, skip

from nets.models.recommender.multitask import TwoTowerMultiTask, \
    ListwiseTwoTowerMultiTask
from nets.layers.recommender import HashEmbedding
from nets.models.mlp import MLP
from nets.utils import get_obj

from nets.tests.integration.models.base import ModelIntegrationABC, \
    RecommenderIntegrationMixin, ListwiseRecommenderIntegrationMixin


class TestTwoTowerMultiTask(RecommenderIntegrationMixin, ModelIntegrationABC, TC):
    """
    """

    temp = os.path.join(os.getcwd(), "multitask-tmp-model")

    def _generate_default_compiled_model(self):
        """
        Instantiate and return a model with the default params and compiled
        with the default loss and optimizer defined in setUp.
        """
        query_model = HashEmbedding(embedding_dim=self._embedding_dim)
        candidate_model = HashEmbedding(embedding_dim=self._embedding_dim)
        target_model = MLP(
                hidden_dims=[2*self._embedding_dim],
                output_dim=1,
                activation="relu",
                output_activation="linear",
                spectral_norm=True
        )

        model = TwoTowerMultiTask(
                target_model=target_model,
                query_model=query_model,
                candidate_model=candidate_model,
                query_id=self._query_id,
                candidate_id=self._candidate_id,
                rank_target=self._rank_target,
                balance=0.5
        )
        model.compile(
            optimizer=get_obj(tf.keras.optimizers, self._optimizer)
        )
        return model


@skip
class TestListwiseTwoTowerMultiTask(ListwiseRecommenderIntegrationMixin, TestTwoTowerMultiTask):
    """
    """

    temp = os.path.join(os.getcwd(), "multitask-tmp-model")

    def _generate_default_compiled_model(self):
        """
        Instantiate and return a model with the default params and compiled
        with the default loss and optimizer defined in setUp.
        """
        query_model = HashEmbedding(embedding_dim=self._embedding_dim)
        candidate_model = HashEmbedding(embedding_dim=self._embedding_dim)

        target_model = MLP(
                hidden_dims=[2*self._embedding_dim],
                output_dim=1,
                activation="relu",
                output_activation="linear",
                spectral_norm=True
        )

        model = ListwiseTwoTowerMultiTask(
                target_model=target_model,
                query_model=query_model,
                candidate_model=candidate_model,
                query_id=self._query_id,
                candidate_id=self._candidate_id,
                rank_target=self._rank_target,
                balance=0.5
        )
        model.compile(
            optimizer=get_obj(tf.keras.optimizers, self._optimizer)
        )
        return model
