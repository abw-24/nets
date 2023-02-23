
from abc import ABCMeta
import tensorflow as tf


@tf.keras.utils.register_keras_serializable("nets")
class BaseModel(tf.keras.Model, metaclass=ABCMeta):
    """
    Abstract base class for `nets` models. Inherits from tf.keras.Model.

    The basic principle of the model implementations is to keep them
    simple, retaining access/use of the high-level APIs whenever possible.
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
        super(BaseModel, self).__init__(name=name, **kwargs)
