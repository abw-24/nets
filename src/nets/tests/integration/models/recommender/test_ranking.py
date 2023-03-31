

import tensorflow as tf
import tensorflow_recommenders as tfrs
import os
from unittest import TestCase as TC

from nets.models.recommender.ranking import TwoTowerRatingsRanking
from nets.layers.recommender import HashEmbedding
from nets.models.mlp import MLP
from nets.utils import get_obj

from nets.tests.utils import try_except_assertion_decorator
from nets.tests.integration.models.base import ModelIntegrationABC, \
    RecommenderIntegrationMixin


class TestTwoTowerRatingsRanking(RecommenderIntegrationMixin, ModelIntegrationABC, TC):
    """
    Fine tuning tester. For simplicity, here we simply create
    """

    temp = os.path.join(os.getcwd(), "rankingtwotower-tmp-model")

    def _generate_default_compiled_model(self):
        """
        Instantiate and return a model with the default params and compiled
        with the default loss and optimizer defined in setUp.
        """
        user_model = HashEmbedding(embedding_dim=self._embedding_dim)
        item_model = HashEmbedding(embedding_dim=self._embedding_dim)
        ratings_model = MLP(
                hidden_dims=[2*self._embedding_dim],
                output_dim=1,
                activation="relu",
                output_activation="linear",
                spectral_norm=True
        )

        model = TwoTowerRatingsRanking(
                ratings_model=ratings_model,
                user_model=user_model,
                item_model=item_model,
                user_features=self._user_features,
                item_features=self._item_features,
                ratings_label=self._ratings_label,
        )
        model.compile(
            optimizer=get_obj(tf.keras.optimizers, self._optimizer)
        )
        return model

