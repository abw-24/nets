
import tensorflow as tf
import tensorflow_recommenders as tfrs
import tensorflow_ranking as tfr

from .base import TwoTowerABC


@tf.keras.utils.register_keras_serializable("nets")
class TwoTowerRetrieval(TwoTowerABC):
    """
    """
    def __init__(self, query_model, candidate_model, query_id,
                 candidate_id, name="TwoTowerRetrieval"):

        super().__init__(name=name)

        self._task = tfrs.tasks.Retrieval()

        self._query_model = query_model
        self._candidate_model = candidate_model
        self._query_id = query_id
        self._candidate_id = candidate_id

    def call(self, inputs):
        """
        """
        query_embeddings = self._query_model(inputs[self._query_id])
        candidate_embeddings = self._candidate_model(inputs[self._candidate_id])
        return query_embeddings, candidate_embeddings

    @tf.function
    def compute_loss(self, features, training=False):
        """
        Compute loss for a batch by invoking the task.
        :param features: Feature batch
        :param training: Training flag
        :return: Loss dictionary
        """
        query_embedding, candidate_embedding = self.__call__(features)
        return self._task.__call__(
                query_embeddings=query_embedding,
                candidate_embeddings=candidate_embedding,
                compute_metrics=False,
                compute_batch_metrics=False
        )
