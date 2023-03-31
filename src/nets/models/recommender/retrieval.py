
import tensorflow as tf
import tensorflow_recommenders as tfrs

from .base import TwoTowerABC


@tf.keras.utils.register_keras_serializable("nets")
class TwoTowerRetrieval(TwoTowerABC):
    """
    The basic two tower model takes pre-defined and/or pre-trained models
    for both the user and item models.
    """
    def __init__(self, user_model, item_model, user_features,
                 item_features, name="TwoTowerRetrieval"):

        super().__init__(name=name)

        self._task = tfrs.tasks.Retrieval()

        self._user_model = user_model
        self._item_model = item_model
        self._user_features = user_features
        self._item_features = item_features

    def compute_loss(self, features, training=False):
        """
        Compute loss for a batch by invoking the task.
        :param features: Feature batch
        :param training: Training flag
        :return: Loss dictionary
        """
        user_embeddings = self.user_model(features[self.user_features])
        item_embeddings = self.item_model(features[self.item_features])

        return self._task(user_embeddings, item_embeddings)