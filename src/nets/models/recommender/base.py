
from abc import ABCMeta
import tensorflow as tf

from nets.models.base import BaseTFRecommenderModel


@tf.keras.utils.register_keras_serializable("nets")
class TwoTowerABC(BaseTFRecommenderModel, metaclass=ABCMeta):

    def __init__(self, name="TwoTowerABC"):
        super().__init__(name=name)

    @property
    def user_id(self):
        return self._user_id

    @property
    def item_id(self):
        return self._item_id

    @property
    def user_model(self):
        return self._user_model

    @property
    def item_model(self):
        return self._item_model
