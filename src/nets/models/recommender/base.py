
from abc import ABCMeta
import tensorflow as tf

from nets.models.base import BaseTFRecommenderModel


@tf.keras.utils.register_keras_serializable("nets")
class TwoTowerABC(BaseTFRecommenderModel, metaclass=ABCMeta):

    def __init__(self, name="TwoTowerABC"):
        super().__init__(name=name)

    @property
    def user_features(self):
        return self._user_features

    @property
    def item_features(self):
        return self._item_features

    @property
    def user_model(self):
        return self._user_model

    @property
    def item_model(self):
        return self._item_model


