
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

    def compute_loss(self, features, training=False):
        """
        Compute loss for a batch by invoking the task.
        :param features: Feature batch
        :param training: Training flag
        :return: Loss dictionary
        """
        query_embedding, candidate_embedding = self.__call__(features)
        return self._task(query_embedding, candidate_embedding)


@tf.keras.utils.register_keras_serializable("nets")
class ListwiseTwoTowerRetrieval(TwoTowerRetrieval):

    """
    """
    def __init__(self, query_model, candidate_model, query_id,
                 candidate_id, name="ListwiseTwoTowerRetrieval"):

        super().__init__(
                query_model=query_model,
                candidate_model=candidate_model,
                query_id=query_id,
                candidate_id=candidate_id,
                name=name
        )

        self._task = tfrs.tasks.Retrieval(
                loss = tfr.keras.losses.ListMLELoss()
        )

    def call(self, inputs):
        """
        """
        query_embedding = self._query_model(inputs[self._query_id])
        candidate_embedding = self._candidate_model(inputs[self._candidate_id])

        list_length = inputs[self._candidate_id].shape[1]
        query_embedding_vec = tf.repeat(
                tf.expand_dims(query_embedding, 1), [list_length], axis=1
        )

        return (query_embedding_vec, candidate_embedding)

    def compute_loss(self, features, training=False):
        """
        """
        query_embedding, candidate_embedding = self.__call__(features)
        return self._task(query_embedding, candidate_embedding)
