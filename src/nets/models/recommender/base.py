
from abc import ABCMeta, abstractmethod
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

    @abstractmethod
    def call(self, inputs, training=False):
        raise NotImplementedError("Abstract")

    @abstractmethod
    def compute_loss(self, features, training=False):
        raise NotImplementedError("Abstract")


class TwoTowerTrait(object):
    """
    Simplifying utilities for generic calls to query/candidate
    embedding models with optional context.
    """

    # To be overwritten / implemented in mixin classes
    _query_model = None
    _query_context_tensor_flag = None
    _query_context_features = None
    _query_id = None
    _candidate_model = None
    _candidate_context_tensor_flag = None
    _candidate_context_features = None
    _candidate_id = None

    def _query_model_with_context(self, inputs):
        """
        Form (embedding_id, context) tuples and pass to the _query_model
        """

        if self._query_context_tensor_flag:
            query_context = inputs[self._query_context_features]
        else:
            query_context = None

        query_embeddings = self._query_model.__call__(
                    (inputs[self._query_id], query_context)
        )

        return query_embeddings

    def _candidate_model_with_context(self, inputs):
        """
        Form (embedding_id, context) tuples and pass to the _candidate_model
        """

        if self._candidate_context_tensor_flag:
            candidate_context = inputs[self._candidate_context_features]
        else:
            candidate_context = None

        candidate_embeddings = self._candidate_model.__call__(
                    (inputs[self._candidate_id], candidate_context)
        )

        return candidate_embeddings