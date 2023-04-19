
import tensorflow as tf
import tensorflow_recommenders as tfrs
import tensorflow_ranking as tfr

from nets.models.recommender.ranking import TwoTowerRanking, \
    ListwiseTwoTowerRanking


@tf.keras.utils.register_keras_serializable("nets")
class TwoTowerMultiTask(TwoTowerRanking):

    """
    Basic pointwise two tower multitask. Balance between retrieval and ranking
     losses is controlled by the `balance` parameter -- the closer to 1, the
    more the ranking loss is favored.

    Assumes `query_model` and `candidate_model` outputs are of the same shape.
    """
    def __init__(self, target_model, query_model, candidate_model, rank_target,
                 query_id, candidate_id, balance=0.5, context_model=None,
                 context_features=None, query_context_features=None,
                 candidate_context_features=None, name="TwoTowerMultiTask"):

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

        self._balance = balance

        self._rank_weight = self._balance
        self._retrieval_weight = 1.0 - self._balance

        self._retrieval_task = tfrs.tasks.Retrieval()

        self._rank_task = tfrs.tasks.Ranking(
                loss = tf.keras.losses.MeanSquaredError(),
                metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )

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
                predictions=scores,
                compute_metrics=not training
        )

        return (self._rank_weight * rank_loss
                + self._retrieval_weight * retrieval_loss)


@tf.keras.utils.register_keras_serializable("nets")
class ListwiseTwoTowerMultiTask(ListwiseTwoTowerRanking):

    """
    Listwise two tower multitask.

    Assumes `query_model` outputs are of shape (batch_size, ..., model_dim),
    while `candidate_model` outputs are of shape (batch_size, n_candidates,
    ..., model_dim). `query_model` embeddings are duplicated and concatenated with
    each `candidate_model` embedding to complete the ranking model inputs.
    """

    def __init__(self, target_model, query_model, candidate_model, rank_target,
                 query_id, candidate_id, balance=0.5, context_model=None,
                 context_features=None, query_context_features=None,
                 candidate_context_features=None,
                 name="ListwiseTwoTowerMultiTask"):

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

        self._balance = balance

        self._rank_weight = self._balance
        self._retrieval_weight = 1.0 - self._balance

        self._retrieval_task = tfrs.tasks.Retrieval()

        self._rank_task = tfrs.tasks.Ranking(
                loss=tfr.keras.losses.ListMLELoss(),
                metrics=[tfr.keras.metrics.NDCGMetric(name="NDCG")]
        )

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