
import tensorflow as tf

from nets.models.base import BaseTFRecommenderModel
from nets.layers.recommender import StringEmbedding


@tf.keras.utils.register_keras_serializable("nets")
class TwoTowerMixin(BaseTFRecommenderModel):

    def __init__(self, name="DenseBlockEmbeddingModel"):
        super().__init__(name=name)

    @property
    def user_features(self):
        return self._user_features

    @property
    def item_features(self):
        return self._item_features

    @property
    def user_model(self):
        return self._user_model

    @property
    def item_model(self):
        return self._item_model

    @property
    def task(self):
        return self._task

    def compute_loss(self, features, training=False):
        """
        Compute loss for a batch by invoking the task.
        :param features: Feature batch
        :param training: Training flag
        :return: Loss dictionary
        """
        user_embeddings = self.user_model(features[self.user_features])
        item_embeddings = self.item_model(features[self.item_features])

        return self.task(user_embeddings, item_embeddings)


@tf.keras.utils.register_keras_serializable("nets")
class SimpleEmbeddingTwoTower(TwoTowerMixin):

    def __init__(self, task, embedding_dim, users, items, user_features,
                 item_features, name="SimpleEmbeddedTwoTower"):
        super().__init__(name=name)

        self._task = task
        self._embedding_dim = embedding_dim
        self._users = users
        self._items = items
        self._user_features = user_features
        self._item_features = item_features

        self._item_model = StringEmbedding(
            vocab=self._items,
            embedding_dim=self._embedding_dim
        )

        self._user_model = StringEmbedding(
            vocab=self._users,
            embedding_dim=self._embedding_dim
        )


@tf.keras.utils.register_keras_serializable("nets")
class FineTuningTwoTower(TwoTowerMixin):
    """
    The fine-tuning two tower model takes pre-defined and pre-trained models
    for both the user and item models.
    """
    def __init__(self, task, user_model, item_model, user_features,
                 item_features, name="SimpleEmbeddedTwoTower"):

        super().__init__(name=name)

        self._user_model = user_model
        self._item_model = item_model
        self._task = task
        self._user_features = user_features
        self._item_features = item_features
