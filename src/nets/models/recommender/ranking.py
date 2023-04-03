
import tensorflow as tf
import tensorflow_recommenders as tfrs
import tensorflow_ranking as tfr

from .base import TwoTowerABC


@tf.keras.utils.register_keras_serializable("nets")
class TwoTowerRanking(TwoTowerABC):

    """
    """
    def __init__(self, target_model, user_model, item_model, rank_target,
                 user_id, item_id, name="TwoTowerRanking"):

        super().__init__(name=name)

        self._target_model = target_model
        self._user_model = user_model
        self._item_model = item_model
        self._rank_target = rank_target
        self._user_id = user_id
        self._item_id = item_id

        # Basic task, can be overwritten / paramterized as needed
        self._task = tfrs.tasks.Ranking(
                loss = tf.keras.losses.MeanSquaredError(),
                metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )

    def call(self, inputs):

        user_embedding = self._user_model(inputs[self._user_id])
        item_embedding = self._item_model(inputs[self._item_id])
        return self._target_model.__call__(tf.concat(
                values=[user_embedding, item_embedding], axis=1
        ))

    def compute_loss(self, features, training=False):

        labels = features.pop(self._rank_target)
        rating_predictions = self.__call__(features)
        return self._task(labels=labels, predictions=rating_predictions)

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
    def __init__(self, target_model, user_model, item_model, rank_target,
                 user_id, item_id, name="ListwiseTwoTowerRanking"):

        super().__init__(
                target_model=target_model,
                user_model=user_model,
                item_model=item_model,
                rank_target=rank_target,
                user_id=user_id,
                item_id=item_id,
                name=name
        )

        self._task = tfrs.tasks.Ranking(
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

        return self._target_model.__call__(concatenated_embeddings)

    def compute_loss(self, features, training=False):

        labels = features.pop(self._rank_target)
        scores = self.__call__(features)

        return self._task(
                labels=labels,
                predictions=tf.squeeze(scores, axis=-1),
        )
