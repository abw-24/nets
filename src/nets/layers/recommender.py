

import tensorflow as tf


@tf.keras.utils.register_keras_serializable("nets")
class StringEmbedding(tf.keras.layers.Layer):

    def __init__(self, max_tokens, embedding_dim=32, name="StringEmbedding"):

        super().__init__(name=name)

        self._max_tokens = max_tokens
        self._embedding_dim = embedding_dim

        self._lookup = tf.keras.layers.StringLookup(
                vocabulary=max_tokens, mask_token=None
        )
        self._embed = tf.keras.layers.Embedding(
                max_tokens + 1, embedding_dim
        )

    def call(self, inputs, training=False):
        return self._embed.__call__(self._lookup.__call__(inputs))

    def get_config(self):
        config = super(StringEmbedding, self).get_config()
        config.update({
            "max_tokens": self._max_tokens,
            "embedding_dim": self._embedding_dim
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)