
from abc import ABCMeta
import tensorflow as tf
import tensorflow_recommenders as tfrs


@tf.keras.utils.register_keras_serializable("nets")
class BaseTFKerasModel(tf.keras.Model, metaclass=ABCMeta):
    """
    Abstract base class for models inheriting from tf.keras.Model.

    The basic principle of the model implementations is to keep them
    simple, retaining use of the high-level APIs whenever possible.
    To that end, the default approach is to override the `train_step` to
    control fitting, and to house tensor computations in tf.keras.Model
    subblocks if reasonable.
    """

    def __init__(self, name: str, **kwargs):
        """

        :param name: Model name (str)
        :param kwargs: Keras model keyword arguments
        :return: None
        """
        super().__init__(name=name, **kwargs)


@tf.keras.utils.register_keras_serializable("nets")
class BaseTFRecommenderModel(tfrs.models.Model, metaclass=ABCMeta):
    """
    Abstract base class for models inheriting from tfrs.models.Model.

    The basic principle of the model implementations is to keep them
    simple, retaining use of the high-level APIs whenever possible.
    """

    def __init__(self, name: str, **kwargs):
        """

        :param name: Model name (str)
        :param kwargs: Recommender keyword arguments
        :return: None
        """
        super().__init__(name=name, **kwargs)