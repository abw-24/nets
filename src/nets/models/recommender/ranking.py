
import tensorflow as tf
import tensorflow_recommenders as tfrs
import tensorflow_ranking as tfr

from .base import TwoTowerABC, TwoTowerTrait


@tf.keras.utils.register_keras_serializable("nets")
class TwoTowerRanking(TwoTowerABC, TwoTowerTrait):

    """
    """
    def __init__(self, target_model, query_model, candidate_model, rank_target,
                 query_id, candidate_id, query_context_features=None,
                 candidate_context_features=None, name="TwoTowerRanking"):

        super().__init__(name=name)

        self._target_model = target_model
        self._query_model = query_model
        self._candidate_model = candidate_model
        self._rank_target = rank_target
        self._query_id = query_id
        self._candidate_id = candidate_id
        self._query_context_features = query_context_features
        self._candidate_context_features = candidate_context_features

        self._query_context_tensor_flag = \
            self._query_context_features is not None
        self._candidate_context_tensor_flag = \
            self._candidate_context_features is not None

        # Basic task, can be overwritten / paramterized as needed
        self._task = tfrs.tasks.Ranking(
                loss = tf.keras.losses.MeanSquaredError(),
                metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )

    def call(self, inputs, training=True):

        query_embeddings = self._query_model_with_context(inputs)
        candidate_embeddings = self._candidate_model_with_context(inputs)

        return self._target_model.__call__(tf.concat(
                values=[query_embeddings, candidate_embeddings], axis=1
        ))

    @tf.function
    def compute_loss(self, features, training=False):

        labels = features[self._rank_target]
        scores = self.__call__(features)
        return self._task.__call__(
                labels=labels,
                predictions=scores,
                compute_metrics=False
        )

    @property
    def target_model(self):
        return self._target_model

    @property
    def rank_target(self):
        return self._rank_target


@tf.keras.utils.register_keras_serializable("nets")
class ListwiseTwoTowerRanking(TwoTowerRanking):

    """
    """
    def __init__(self, target_model, query_model, candidate_model, rank_target,
                 query_id, candidate_id, query_context_features=None,
                 candidate_context_features=None, name="ListwiseTwoTowerRanking"):

        super().__init__(
                target_model=target_model,
                query_model=query_model,
                candidate_model=candidate_model,
                rank_target=rank_target,
                query_id=query_id,
                candidate_id=candidate_id,
                query_context_features=query_context_features,
                candidate_context_features=candidate_context_features,
                name=name
        )

        self._task = tfrs.tasks.Ranking(
                loss = tfr.keras.losses.ListMLELoss(),
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

        return self._target_model.__call__(concatenated_embeddings)

    @tf.function
    def compute_loss(self, features, training=False):

        labels = features[self._rank_target]
        scores = self.__call__(features)

        return self._task.__call__(
                labels=labels,
                predictions=tf.squeeze(scores, axis=-1),
                compute_metrics=False
        )
