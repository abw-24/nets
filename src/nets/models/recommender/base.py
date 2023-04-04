
from abc import ABCMeta
import tensorflow as tf

from nets.models.base import BaseTFRecommenderModel


@tf.keras.utils.register_keras_serializable("nets")
class TwoTowerABC(BaseTFRecommenderModel, metaclass=ABCMeta):

    def __init__(self, name="TwoTowerABC"):
        super().__init__(name=name)

    @property
    def query_id(self):
        return self._query_id

    @property
    def candidate_id(self):
        return self._candidate_id

    @property
    def query_model(self):
        return self._query_model

    @property
    def candidate_model(self):
        return self._candidate_model
