
import tensorflow as tf
import tensorflow_recommenders as tfrs
import tensorflow_ranking as tfr

from .base import TwoTowerABC


@tf.keras.utils.register_keras_serializable("nets")
class TwoTowerMultiTask(TwoTowerABC):

    """
    """
    def __init__(self, target_model, query_model, candidate_model, rank_target,
                 query_id, candidate_id, balance=0.5, name="TwoTowerRanking"):

        super().__init__(name=name)

        self._target_model = target_model
        self._query_model = query_model
        self._candidate_model = candidate_model
        self._query_id = query_id
        self._candidate_id = candidate_id
        self._rank_target = rank_target
        self._balance = balance

        self._rank_weight = self._balance
        self._retrieval_weight = 1.0 - self._balance

        self._retrieval_task = tfrs.tasks.Retrieval()

        self._rank_task = tfrs.tasks.Ranking(
                loss = tf.keras.losses.MeanSquaredError(),
                metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )

    def call(self, inputs):

        query_embedding = self._query_model(inputs[self._query_id])
        candidate_embedding = self._candidate_model(inputs[self._candidate_id])
        rank_prediction = self._target_model.__call__(tf.concat(
                values=[query_embedding, candidate_embedding], axis=1
        ))

        return (query_embedding, candidate_embedding, rank_prediction)

    def compute_loss(self, features, training=False):

        labels = features.pop(self._rank_target)
        query_embeddings, candidate_embeddings, rating_predictions = self.__call__(features)

        retrieval_loss = self._retrieval_task(query_embeddings, candidate_embeddings)
        rating_loss = self._rank_task(labels=labels, predictions=rating_predictions)

        return (self._rank_weight * rating_loss
                + self._retrieval_weight * retrieval_loss)


@tf.keras.utils.register_keras_serializable("nets")
class ListwiseTwoTowerMultiTask(TwoTowerMultiTask):

    """
    """
    def __init__(self, target_model, query_model, candidate_model, rank_target,
                 query_id, candidate_id, balance=0.5, name="ListwiseTwoTowerMultiTask"):

        super().__init__(
                target_model=target_model,
                query_model=query_model,
                candidate_model=candidate_model,
                rank_target=rank_target,
                query_id=query_id,
                candidate_id=candidate_id,
                balance=balance,
                name=name
        )

        self._retrieval_task = tfrs.tasks.Retrieval(
                loss = tfr.keras.losses.ListMLELoss()
        )

        self._rank_task = tfrs.tasks.Ranking(
                loss = tfr.keras.losses.ListMLELoss(),
                metrics=[tfr.keras.metrics.NDCGMetric(name="NDCG")]
        )

    def call(self, inputs):

        query_embedding = self._query_model(inputs[self._query_id])
        candidate_embedding = self._candidate_model(inputs[self._candidate_id])

        list_length = inputs[self._candidate_id].shape[1]
        query_embedding_vec = tf.repeat(
                tf.expand_dims(query_embedding, 1), [list_length], axis=1
        )

        concatenated_embeddings = tf.concat(
                [query_embedding_vec, candidate_embedding], 2
        )

        rank_prediction = self._target_model.__call__(concatenated_embeddings)

        return (query_embedding, candidate_embedding, rank_prediction)

    def compute_loss(self, features, training=False):

        labels = features.pop(self._rank_target)
        query_embedding, candidate_embedding, scores = self.__call__(features)

        retrieval_loss = self._retrieval_task(query_embedding, candidate_embedding)
        rank_loss = self._rank_task(
                labels=labels,
                predictions=tf.squeeze(scores, axis=-1),
        )

        return (self._rank_weight * rank_loss
                + self._retrieval_weight * retrieval_loss)
















































