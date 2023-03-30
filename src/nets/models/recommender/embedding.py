
import tensorflow as tf

from nets.models.base import BaseTFKerasModel
from nets.layers.recommender import HashEmbedding
from nets.layers.dense import DenseBlock


@tf.keras.utils.register_keras_serializable("nets")
class DeepHashEmbedding(BaseTFKerasModel):

    _dense_config = {
        "hidden_dims": [32],
        "activation": "relu",
        "spectral_norm": True
    }

    def __init__(self, hash_bins=200000, hash_embedding_dim=64,
                 embedding_dim=16, name="DeepHashEmbedding", **kwargs):

        super().__init__(name=name)

        self._dense_config.update(kwargs)

        self._hash_bins = hash_bins
        self._hash_embedding_dim = hash_embedding_dim
        self._embedding_dim = embedding_dim

        self._embedding = HashEmbedding(
                hash_bins=self._hash_bins, embedding_dim=self._hash_embedding_dim
        )
        self._dense_block = DenseBlock.from_config(
                self._dense_config
        )
        self._final_layer = tf.keras.layers.Dense(
                units=self._embedding_dim, activation="linear"
        )

    def call(self, inputs):

        return self._final_layer.__call__(
                self._dense_block.__call__(
                        self._embedding.__call__(inputs)
                )
        )

    def get_config(self):
        config = super(DeepHashEmbedding, self).get_config()
        config.update({
            "hash_bins": self._hash_bins,
            "hash_embedding_dim": self._hash_embedding_dim,
            "embedding_dim": self._embedding_dim
        })
        config.update(self._dense_config)
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)