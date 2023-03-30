
import tensorflow as tf
import tensorflow_recommenders as tfrs

from nets.models.base import BaseTFRecommenderModel
from nets.layers.recommender import StringEmbedding


@tf.keras.utils.register_keras_serializable("nets")
class TwoTowerMixin(object):

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
class TwoTowerRetrieval(BaseTFRecommenderModel, TwoTowerMixin):
    """
    The basic two tower retrieval model takes pre-defined (and possibly
    pre-trained) models for both the user and item.
    """
    def __init__(self, user_model, item_model, user_features,
                 item_features, name="TwoTowerRetrieval"):

        super().__init__(name=name)

        self._task = tfrs.tasks.Retrieval()

        self._user_model = user_model
        self._item_model = item_model
        self._user_features = user_features
        self._item_features = item_features


@tf.keras.utils.register_keras_serializable("nets")
class SimpleEmbeddingTwoTowerRetrieval(TwoTowerRetrieval):

    def __init__(self, embedding_dim, users, items, user_features,
                 item_features, name="SimpleEmbeddedTwoTowerRetrieval"):

        self._embedding_dim = embedding_dim
        self._users = users
        self._items = items

        user_model = StringEmbedding(
            vocab=self._users, embedding_dim=self._embedding_dim
        )

        item_model = StringEmbedding(
            vocab=self._items, embedding_dim=self._embedding_dim
        )

        super().__init__(
            user_model=user_model,
            item_model=item_model,
            user_features=user_features,
            item_features=item_features,
            name=name
        )


@tf.keras.utils.register_keras_serializable("nets")
class TwoTowerRatingsRanking(BaseTFRecommenderModel, TwoTowerMixin):

    """
    The two tower ratings ranking model requires a ratings model in addition
    to the user and item models (all pre-defined, and possible pre-trained).
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

        user_features, item_features = inputs
        user_embedding = self.user_model(user_features)
        item_embedding = self.item_model(item_features)

        return self._ratings_model.__call__(tf.concat(
                values=[user_embedding, item_embedding], axis=1
        ))

    def compute_loss(self, features, training=False):

        labels = features.pop(self._ratings_label)
        feature_tuple = (
            features[self.user_features], features[self.item_features]
        )
        rating_predictions = self.__call__(feature_tuple)
        return self.task(labels=labels, predictions=rating_predictions)

    @property
    def ratings_model(self):
        return self._ratings_model

    @property
    def ratings_label(self):
        return self._ratings_label
