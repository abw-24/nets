

import tensorflow as tf
import tensorflow_recommenders as tfrs

from .base import TwoTowerABC


@tf.keras.utils.register_keras_serializable("nets")
class TwoTowerRatingsRanking(TwoTowerABC):

    """
    The fine-tuning two tower model takes pre-defined and pre-trained models
    for both the user and item models.
    """
    def __init__(self, ratings_model, user_model, item_model, user_features,
                 item_features, ratings_label, name="TwoTowerRatingsRanking"):

        super().__init__(name=name)

        self._ratings_model = ratings_model
        self._user_model = user_model
        self._item_model = item_model
        self._user_features = user_features
        self._item_features = item_features
        self._ratings_label = ratings_label

        # Basic task, can be overwritten / paramterized as needed
        self._task = tfrs.tasks.Ranking(
                loss = tf.keras.losses.MeanSquaredError(),
                metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )

    def call(self, inputs):

        user_embedding = self.user_model(inputs[self.user_features])
        item_embedding = self.item_model(inputs[self.item_features])
        return self._ratings_model.__call__(tf.concat(
                values=[user_embedding, item_embedding], axis=1
        ))

    def compute_loss(self, features, training=False):

        labels = features.pop(self._ratings_label)
        rating_predictions = self.__call__(features)
        return self._task(labels=labels, predictions=rating_predictions)
