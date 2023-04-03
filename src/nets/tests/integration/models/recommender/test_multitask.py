

import tensorflow as tf
import os
from unittest import TestCase as TC

from nets.models.recommender.multitask import TwoTowerMultiTask, \
    ListwiseTwoTowerMultiTask
from nets.layers.recommender import HashEmbedding
from nets.models.mlp import MLP
from nets.utils import get_obj

from nets.tests.integration.models.base import ModelIntegrationABC, \
    RecommenderIntegrationMixin, ListwiseRecommenderIntegrationMixin


class TestTwoTowerMultiTask(RecommenderIntegrationMixin, ModelIntegrationABC, TC):
    """
    Fine tuning tester. For simplicity, here we simply create
    """

    temp = os.path.join(os.getcwd(), "multitask-tmp-model")

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

        model = TwoTowerMultiTask(
                target_model=target_model,
                user_model=user_model,
                item_model=item_model,
                user_id=self._user_id,
                item_id=self._item_id,
                rank_target=self._rank_target,
                balance=0.5
        )
        model.compile(
            optimizer=get_obj(tf.keras.optimizers, self._optimizer)
        )
        return model


class TestListwiseTwoTowerMultiTask(ListwiseRecommenderIntegrationMixin, TestTwoTowerMultiTask):
    """
    Fine tuning tester. For simplicity, here we simply create
    """

    temp = os.path.join(os.getcwd(), "multitask-tmp-model")

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

        model = ListwiseTwoTowerMultiTask(
                target_model=target_model,
                user_model=user_model,
                item_model=item_model,
                user_id=self._user_id,
                item_id=self._item_id,
                rank_target=self._rank_target,
                balance=0.5
        )
        model.compile(
            optimizer=get_obj(tf.keras.optimizers, self._optimizer)
        )
        return model
