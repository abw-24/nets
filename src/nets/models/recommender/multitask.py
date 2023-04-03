
import tensorflow as tf
import tensorflow_recommenders as tfrs
import tensorflow_ranking as tfr

from .base import TwoTowerABC


@tf.keras.utils.register_keras_serializable("nets")
class TwoTowerMultiTask(TwoTowerABC):

    """
    """
    def __init__(self, target_model, user_model, item_model, rank_target,
                 user_id, item_id, balance=0.5, name="TwoTowerRanking"):

        super().__init__(name=name)

        self._target_model = target_model
        self._user_model = user_model
        self._item_model = item_model
        self._user_id = user_id
        self._item_id = item_id
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

        user_embedding = self._user_model(inputs[self._user_id])
        item_embedding = self._item_model(inputs[self._item_id])
        rank_prediction = self._target_model.__call__(tf.concat(
                values=[user_embedding, item_embedding], axis=1
        ))

        return (user_embedding, item_embedding, rank_prediction)

    def compute_loss(self, features, training=False):

        labels = features.pop(self._rank_target)
        user_embeddings, item_embeddings, rating_predictions = self.__call__(features)

        retrieval_loss = self._retrieval_task(user_embeddings, item_embeddings)
        rating_loss = self._rank_task(labels=labels, predictions=rating_predictions)

        return (self._rank_weight * rating_loss
                + self._retrieval_weight * retrieval_loss)


@tf.keras.utils.register_keras_serializable("nets")
class ListwiseTwoTowerMultiTask(TwoTowerMultiTask):

    """
    """
    def __init__(self, target_model, user_model, item_model, rank_target,
                 user_id, item_id, balance=0.5, name="ListwiseTwoTowerMultiTask"):

        super().__init__(
                target_model=target_model,
                user_model=user_model,
                item_model=item_model,
                rank_target=rank_target,
                user_id=user_id,
                item_id=item_id,
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

        user_embedding = self._user_model(inputs[self._user_id])
        item_embedding = self._item_model(inputs[self._item_id])

        list_length = inputs[self._item_id].shape[1]
        user_embedding_vec = tf.repeat(
                tf.expand_dims(user_embedding, 1), [list_length], axis=1
        )

        concatenated_embeddings = tf.concat(
                [user_embedding_vec, item_embedding], 2
        )

        rank_prediction = self._target_model.__call__(concatenated_embeddings)

        return (user_embedding, item_embedding, rank_prediction)

    def compute_loss(self, features, training=False):

        labels = features.pop(self._rank_target)
        user_embedding, item_embedding, scores = self.__call__(features)

        retrieval_loss = self._retrieval_task(user_embedding, item_embedding)
        rank_loss = self._rank_task(
                labels=labels,
                predictions=tf.squeeze(scores, axis=-1),
        )

        return (self._rank_weight * rank_loss
                + self._retrieval_weight * retrieval_loss)
















































