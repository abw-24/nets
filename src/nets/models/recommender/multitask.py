
import tensorflow as tf
import tensorflow_recommenders as tfrs
import tensorflow_ranking as tfr

from .base import TwoTowerABC, TwoTowerTrait


@tf.keras.utils.register_keras_serializable("nets")
class TwoTowerMultiTask(TwoTowerABC, TwoTowerTrait):

    """
    """
    def __init__(self, target_model, query_model, candidate_model, rank_target,
                 query_id, candidate_id, balance=0.5, query_context_features=None,
                 candidate_context_features=None, name="TwoTowerMultitask"):

        super().__init__(name=name)

        self._target_model = target_model
        self._query_model = query_model
        self._candidate_model = candidate_model
        self._query_id = query_id
        self._candidate_id = candidate_id
        self._rank_target = rank_target
        self._balance = balance
        self._query_context_features = query_context_features
        self._candidate_context_features = candidate_context_features

        self._query_context_tensor_flag = \
            self._query_context_features is not None
        self._candidate_context_tensor_flag = \
            self._candidate_context_features is not None

        self._rank_weight = self._balance
        self._retrieval_weight = 1.0 - self._balance

        self._retrieval_task = tfrs.tasks.Retrieval()

        self._rank_task = tfrs.tasks.Ranking(
                loss = tf.keras.losses.MeanSquaredError(),
                metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )

    def call(self, inputs, training=True):

        query_embeddings = self._query_model_with_context(inputs)
        candidate_embeddings = self._candidate_model_with_context(inputs)

        rank_prediction = self._target_model.__call__(tf.concat(
                values=[query_embeddings, candidate_embeddings], axis=1
        ))

        return (query_embeddings, candidate_embeddings, rank_prediction)

    @tf.function
    def compute_loss(self, features, training=True):

        labels = features[self._rank_target]
        query_embeddings, candidate_embeddings, scores = self.__call__(features)

        retrieval_loss = self._retrieval_task.__call__(
                query_embeddings=query_embeddings,
                candidate_embeddings=candidate_embeddings,
                compute_metrics=not training
        )
        rank_loss = self._rank_task.__call__(
                labels=labels,
                predictions=tf.squeeze(scores, axis=-1),
                compute_metrics=not training
        )

        return (self._rank_weight * rank_loss
                + self._retrieval_weight * retrieval_loss)


@tf.keras.utils.register_keras_serializable("nets")
class ListwiseTwoTowerMultiTask(TwoTowerMultiTask):

    """
    """
    def __init__(self, target_model, query_model, candidate_model, rank_target,
                 query_id, candidate_id, balance=0.5, query_context_features=None,
                 candidate_context_features=None, name="ListwiseTwoTowerMultitask"):

        super().__init__(
                target_model=target_model,
                query_model=query_model,
                candidate_model=candidate_model,
                rank_target=rank_target,
                query_id=query_id,
                candidate_id=candidate_id,
                balance=balance,
                query_context_features=query_context_features,
                candidate_context_features=candidate_context_features,
                name=name
        )

        self._retrieval_task = tfrs.tasks.Retrieval()

        self._rank_task = tfrs.tasks.Ranking(
                loss=tfr.keras.losses.ListMLELoss(),
                metrics=[tfr.keras.metrics.NDCGMetric(name="NDCG")]
        )

    def call(self, inputs, training=True):

        query_embeddings = self._query_model_with_context(inputs)
        candidate_embeddings = self._candidate_model_with_context(inputs)

        list_length = inputs[self._candidate_id].shape[1]

        query_embedding_vec = tf.repeat(
                tf.expand_dims(query_embeddings, 1), [list_length], axis=1
        )

        concatenated_embeddings = tf.concat(
                [query_embedding_vec, candidate_embeddings], 2
        )

        rank_prediction = self._target_model.__call__(concatenated_embeddings)

        return (query_embedding_vec, candidate_embeddings, rank_prediction)

    @tf.function
    def compute_loss(self, features, training=True):

        labels = features[self._rank_target]
        query_embeddings, candidate_embeddings, scores = self.__call__(features)

        retrieval_loss = self._retrieval_task.__call__(
                query_embeddings=query_embeddings,
                candidate_embeddings=candidate_embeddings,
                compute_metrics=not training
        )
        rank_loss = self._rank_task.__call__(
                labels=labels,
                predictions=tf.squeeze(scores, axis=-1),
                compute_metrics=not training
        )

        return (self._rank_weight * rank_loss
                + self._retrieval_weight * retrieval_loss)

