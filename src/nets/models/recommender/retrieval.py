
import tensorflow as tf
import tensorflow_recommenders as tfrs
import tensorflow_ranking as tfr

from .base import TwoTowerABC


@tf.keras.utils.register_keras_serializable("nets")
class TwoTowerRetrieval(TwoTowerABC):
    """
    """
    def __init__(self, user_model, item_model, user_id,
                 item_id, name="TwoTowerRetrieval"):

        super().__init__(name=name)

        self._task = tfrs.tasks.Retrieval()

        self._user_model = user_model
        self._item_model = item_model
        self._user_id = user_id
        self._item_id = item_id

    def call(self, inputs):
        """

        :param inputs:
        :return:
        """
        user_embeddings = self._user_model(inputs[self._user_id])
        item_embeddings = self._item_model(inputs[self._item_id])
        return user_embeddings, item_embeddings

    def compute_loss(self, features, training=False):
        """
        Compute loss for a batch by invoking the task.
        :param features: Feature batch
        :param training: Training flag
        :return: Loss dictionary
        """
        user_embedding, item_embedding = self.__call__(features)
        return self._task(user_embedding, item_embedding)


@tf.keras.utils.register_keras_serializable("nets")
class ListwiseTwoTowerRetrieval(TwoTowerRetrieval):

    """
    """
    def __init__(self, user_model, item_model, user_id,
                 item_id, name="ListwiseTwoTowerRetrieval"):

        super().__init__(
                user_model=user_model,
                item_model=item_model,
                user_id=user_id,
                item_id=item_id,
                name=name
        )

        self._task = tfrs.tasks.Retrieval(
                loss = tfr.keras.losses.ListMLELoss()
        )

    def call(self, inputs):

        user_embedding = self._user_model(inputs[self._user_id])
        item_embedding = self._item_model(inputs[self._item_id])

        list_length = inputs[self._item_id].shape[1]
        user_embedding_vec = tf.repeat(
                tf.expand_dims(user_embedding, 1), [list_length], axis=1
        )

        return (user_embedding_vec, item_embedding)

    def compute_loss(self, features, training=False):

        user_embedding, item_embedding = self.__call__(features)

        return self._task(user_embedding, item_embedding)
