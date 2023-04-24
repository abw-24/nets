
from abc import ABCMeta, abstractmethod
import tensorflow as tf
import tensorflow_recommenders as tfrs


@tf.keras.utils.register_keras_serializable("nets")
class BaseTFKerasModel(tf.keras.Model, metaclass=ABCMeta):
    """
    Abstract base class for models inheriting from tf.keras.Model.

    The basic principle of the tf.keras model implementations is to keep them
    simple, retaining use of the high-level APIs whenever possible.
    To that end, the default approach is to override the `train_step` to
    control fitting, and to house tensor computations in registered
    tf.keras.Model subblocks and/or registered custom layers.
    """

    def __init__(self, name: str, **kwargs):
        """

        :param name: Model name (str)
        :param kwargs: Keras model keyword arguments
        :return: None
        """
        super().__init__(name=name, **kwargs)

    @abstractmethod
    def call(self, inputs, training=False):
        raise NotImplementedError("Abstract.")

    @abstractmethod
    def get_config(self):
        raise NotImplementedError("Abstract.")


@tf.keras.utils.register_keras_serializable("nets")
class BaseTFRecommenderModel(tfrs.models.Model, metaclass=ABCMeta):
    """
    Abstract base class for models inheriting from tfrs.models.Model.

    The basic principle of the model implementations is to keep them
    simple, retaining use of the high-level APIs whenever possible.

    Generally speaking, only submodels are of interest with recommenders (for
    post-training serialization), so recommender model definitions often
    focus on `compute_loss`. If for some reason the top-level model needs to
    be serialized as a whole for later inference, inherit from BaseTFKerasModel
    and follow that paradigm instead.
    """

    def __init__(self, name: str, **kwargs):
        """

        :param name: Model name (str)
        :param kwargs: Recommender keyword arguments
        :return: None
        """
        super().__init__(name=name, **kwargs)

    @abstractmethod
    def call(self, inputs, training=False):
        raise NotImplementedError("Abstract.")

    @abstractmethod
    def compute_loss(self, features, training=False):
        raise NotImplementedError("Abstract")