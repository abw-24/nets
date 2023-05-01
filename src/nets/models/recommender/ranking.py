import tensorflow as tf
import tensorflow_recommenders as tfrs
import tensorflow_ranking as tfr

from .base import TwoTowerABC, TwoTowerMixin


@tf.keras.utils.register_keras_serializable("nets")
class TwoTowerRanking(TwoTowerABC, TwoTowerMixin):

    """
    Basic pointwise two tower ranking.
    """
    def __init__(self, target_model, query_model, candidate_model, rank_target,
                 query_id, candidate_id, context_model=None, context_features=None,
                 query_context_features=None, candidate_context_features=None,
                 loss=None, name="TwoTowerRanking"):

        super().__init__(name=name)

        self._target_model = target_model
        self._query_model = query_model
        self._candidate_model = candidate_model
        self._rank_target = rank_target
        self._query_id = query_id
        self._candidate_id = candidate_id
        self._context_model = context_model
        self._context_features = context_features
        self._query_context_features = query_context_features
        self._candidate_context_features = candidate_context_features
        self._loss = loss

        self._query_context_flag = \
            self._query_context_features is not None
        self._candidate_context_flag = \
            self._candidate_context_features is not None
        self._context_flag = self._context_features is not None

        # Basic task, can be overwritten / paramterized as needed
        if self._loss is None:
            self._loss = tf.keras.losses.MeanSquaredError()

        self._task = tfrs.tasks.Ranking(loss=self._loss)

    def call(self, inputs, training=True):

        query_embeddings = self._query_model_with_context(inputs)
        candidate_embeddings = self._candidate_model_with_context(inputs)
        # Concatenate the query and candidate embeddings
        concat_embeddings = tf.concat(
                    [query_embeddings, candidate_embeddings], -1
        )

        # General context
        if self._context_flag:
            context_embeddings = self._context_model.__call__(
                    inputs[self._context_features]
            )
            concat_embeddings = tf.concat(
                    [concat_embeddings, context_embeddings], -1
        )

        # Pass complete embedding to target model and return the triplet
        rank_prediction = self._target_model.__call__(concat_embeddings)

        return (query_embeddings, candidate_embeddings, rank_prediction)

    @tf.function
    def compute_loss(self, features, training=True):

        labels = features[self._rank_target]
        _, _, scores = self.__call__(features)

        return self._task.__call__(
                labels=labels,
                predictions=scores,
                compute_metrics=not training
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
    Listwise two tower ranking.

    Assumes `query_model` outputs are of shape (batch_size, ..., model_dim)
    and `candidate_model` outputs are of shape (batch_size, n_candidates,
    ..., model_dim). `query_model` embeddings are copied and concatenated with
    each `candidate_model` embedding to complete the ranking model inputs.
    """
    def __init__(self, target_model, query_model, candidate_model, rank_target,
                 query_id, candidate_id, context_model=None, context_features=None,
                 query_context_features=None, candidate_context_features=None,
                 loss=None, name="TwoTowerRanking"):

        # Notice `loss` is explicitly set to None in the parent constructor
        # call -- handled separately below
        super().__init__(
                target_model=target_model,
                query_model=query_model,
                candidate_model=candidate_model,
                rank_target=rank_target,
                query_id=query_id,
                candidate_id=candidate_id,
                context_model=context_model,
                context_features=context_features,
                query_context_features=query_context_features,
                candidate_context_features=candidate_context_features,
                loss=None,
                name=name
        )

        # Overwrite the parent constructor handling
        self._loss = loss
        if self._loss is None:
            self._loss = tfr.keras.losses.ListMLELoss()

        self._task = tfrs.tasks.Ranking(
                loss=self._loss,
                metrics=[tfr.keras.metrics.NDCGMetric(name="NDCG")]
        )

    def call(self, inputs, training=True):

        query_embeddings = self._query_model_with_context(inputs)
        candidate_embeddings = self._candidate_model_with_context(inputs)

        # Expand on candidates dimension (assumed to be axis=1)
        n_candidates = inputs[self._candidate_id].shape[1]
        query_embedding_vec = tf.repeat(
                tf.expand_dims(query_embeddings, 1), [n_candidates], axis=1
        )

        concat_embeddings = tf.concat(
                [query_embedding_vec, candidate_embeddings], -1
        )

        # General context
        if self._context_flag:
            context_embeddings = self._context_model.__call__(
                    inputs[self._context_features]
            )
            concat_embeddings = tf.concat(
                    [concat_embeddings, context_embeddings], -1
            )

        # Pass complete embedding to target model and return the triplet
        rank_prediction = self._target_model.__call__(concat_embeddings)

        return (query_embedding_vec, candidate_embeddings, rank_prediction)

    @tf.function
    def compute_loss(self, features, training=True):

        labels = features[self._rank_target]
        _, _, scores = self.__call__(features)

        return self._task.__call__(
                labels=labels,
                predictions=tf.squeeze(scores, axis=-1),
                compute_metrics=not training
        )