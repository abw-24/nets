

import tensorflow as tf


@tf.keras.utils.register_keras_serializable("nets")
class StringEmbedding(tf.keras.layers.Layer):

    def __init__(self, vocab, embedding_dim=32, name="StringEmbedding"):

        super().__init__(name=name)

        self._vocab = vocab
        self._embedding_dim = embedding_dim

        self._lookup = tf.keras.layers.StringLookup(
                vocabulary=self._vocab, mask_token=None
        )
        self._embed = tf.keras.layers.Embedding(
                len(self._vocab) + 1, embedding_dim
        )

    def call(self, inputs, training=False):
        return self._embed.__call__(self._lookup.__call__(inputs))

    def get_config(self):
        config = super(StringEmbedding, self).get_config()
        config.update({
            "vocab": self._vocab,
            "embedding_dim": self._embedding_dim
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable("nets")
class HashEmbedding(tf.keras.layers.Layer):

    def __init__(self, hash_bins=262144, embedding_dim=32,
                 name="HashEmbedding"):

        super().__init__(name=name)

        self._hash_bins = hash_bins
        self._embedding_dim = embedding_dim

        self._lookup = tf.keras.layers.Hashing(
                num_bins=self._hash_bins
        )
        self._embed = tf.keras.layers.Embedding(
                self._hash_bins, embedding_dim
        )

    def call(self, inputs, training=False):
        return self._embed.__call__(self._lookup.__call__(inputs))

    def get_config(self):
        config = super(HashEmbedding, self).get_config()
        config.update({
            "hash_bins": self._hash_bins,
            "embedding_dim": self._embedding_dim
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
