

import tensorflow as tf
import tensorflow_recommenders as tfrs
import tensorflow_datasets as tfds
import os
from unittest import TestCase as TC

from nets.models.recommender.ranking import TwoTowerRanking, \
    ListwiseTwoTowerRanking
from nets.layers.recommender import HashEmbedding
from nets.models.mlp import MLP
from nets.utils import get_obj

from nets.tests.integration.models.base import ModelIntegrationABC, \
    RecommenderIntegrationMixin, ListwiseRecommenderIntegrationMixin


class TestTwoTowerRanking(RecommenderIntegrationMixin, ModelIntegrationABC, TC):
    """
    """

    temp = os.path.join(os.getcwd(), "rankingtwotower-tmp-model")

    def _generate_default_compiled_model(self):
        """
        Instantiate and return a model with the default params and compiled
        with the default loss and optimizer defined in setUp.
        """
        user_model = HashEmbedding(embedding_dim=self._embedding_dim)
        item_model = HashEmbedding(embedding_dim=self._embedding_dim)
        target_model = MLP(
                hidden_dims=[2*self._embedding_dim],
                output_dim=1,
                activation="relu",
                output_activation="linear",
                spectral_norm=True
        )

        model = TwoTowerRanking(
                target_model=target_model,
                user_model=user_model,
                item_model=item_model,
                user_id=self._user_id,
                item_id=self._item_id,
                rank_target=self._rank_target,
        )
        model.compile(
            optimizer=get_obj(tf.keras.optimizers, self._optimizer)
        )
        return model


class TestListwiseTwoTowerRanking(ListwiseRecommenderIntegrationMixin, TestTwoTowerRanking):
    """
    """

    temp = os.path.join(os.getcwd(), "rankingtwotower-tmp-model")

    def _generate_default_compiled_model(self):
        """
        Instantiate and return a model with the default params and compiled
        with the default loss and optimizer defined in setUp.
        """
        user_model = HashEmbedding(embedding_dim=self._embedding_dim)
        item_model = HashEmbedding(embedding_dim=self._embedding_dim)
        target_model = MLP(
                hidden_dims=[2*self._embedding_dim],
                output_dim=1,
                activation="relu",
                output_activation="linear",
                spectral_norm=True
        )

        model = ListwiseTwoTowerRanking(
                target_model=target_model,
                user_model=user_model,
                item_model=item_model,
                user_id=self._user_id,
                item_id=self._item_id,
                rank_target=self._rank_target,
        )
        model.compile(
            optimizer=get_obj(tf.keras.optimizers, self._optimizer)
        )
        return model