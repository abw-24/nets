

import tensorflow as tf
import tensorflow_recommenders as tfrs

from .base import TwoTowerABC


@tf.keras.utils.register_keras_serializable("nets")
class TwoTowerMultiTask(TwoTowerABC):

    """
    Simple multitask model to
    """
    def __init__(self, ratings_model, user_model, item_model, user_features,
                 item_features, ratings_label, balance=0.5,
                 name="TwoTowerRatingsRanking"):

        super().__init__(name=name)

        self._ratings_model = ratings_model
        self._user_model = user_model
        self._item_model = item_model
        self._user_features = user_features
        self._item_features = item_features
        self._ratings_label = ratings_label
        self._balance = balance

        self._rating_weight = self._balance
        self._retrieval_weight = 1.0 - self._balance

        self._retrieval_task = tfrs.tasks.Retrieval()

        self._rank_task = tfrs.tasks.Ranking(
                loss = tf.keras.losses.MeanSquaredError(),
                metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )

    def call(self, inputs):

        user_embedding = self.user_model(inputs[self.user_features])
        item_embedding = self.item_model(inputs[self.item_features])
        ratings = self._ratings_model.__call__(tf.concat(
                values=[user_embedding, item_embedding], axis=1
        ))

        return (user_embedding, item_embedding, ratings)

    def compute_loss(self, features, training=False):

        labels = features.pop(self._ratings_label)
        user_embeddings, item_embeddings, rating_predictions = self.__call__(features)

        retrieval_loss = self._retrieval_task(user_embeddings, item_embeddings)
        rating_loss = self._rank_task(labels=labels, predictions=rating_predictions)

        # And combine them using the loss weights.
        return (self._rating_weight * rating_loss
                + self._retrieval_weight * retrieval_loss)
