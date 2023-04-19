
import tensorflow as tf
import tensorflow_recommenders as tfrs

from .base import TwoTowerABC, TwoTowerTrait


@tf.keras.utils.register_keras_serializable("nets")
class TwoTowerRetrieval(TwoTowerTrait, TwoTowerABC):
    """
    """
    def __init__(self, query_model, candidate_model, query_id,
                 candidate_id, query_context_features=None,
                 candidate_context_features=None, name="TwoTowerRetrieval"):

        super().__init__(name=name)

        self._task = tfrs.tasks.Retrieval()

        self._query_model = query_model
        self._candidate_model = candidate_model
        self._query_id = query_id
        self._candidate_id = candidate_id
        self._query_context_features = query_context_features
        self._candidate_context_features = candidate_context_features

        self._query_context_flag = \
            self._query_context_features is not None
        self._candidate_context_flag = \
            self._candidate_context_features is not None

    def call(self, inputs, training=False):
        """
        """
        query_embeddings = self._query_model_with_context(inputs)
        candidate_embeddings = self._candidate_model_with_context(inputs)

        return query_embeddings, candidate_embeddings

    @tf.function
    def compute_loss(self, features, training=True):
        """
        Compute loss for a batch by invoking the task
        """
        query_embeddings, candidate_embeddings = self.__call__(features)
        return self._task.__call__(
                query_embeddings=query_embeddings,
                candidate_embeddings=candidate_embeddings,
                compute_metrics=not training,
                compute_batch_metrics=not training
        )