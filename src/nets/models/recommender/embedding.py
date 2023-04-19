
import tensorflow as tf

from nets.models.base import BaseTFKerasModel
from nets.layers.dense import DenseBlock
from nets.layers.sequence import MultiHeadSelfAttention


@tf.keras.utils.register_keras_serializable("nets")
class StringEmbedding(BaseTFKerasModel):

    def __init__(self, vocab, embedding_dim=32, context_model=None,
                 name="StringEmbedding"):

        super().__init__(name=name)

        self._vocab = vocab
        self._embedding_dim = embedding_dim
        self._context_model = context_model

        # Create a tf constant boolean for checking during call
        self._context_flag = self._context_model is not None

        self._lookup = tf.keras.layers.StringLookup(
                vocabulary=self._vocab, mask_token=None
        )
        self._embed = tf.keras.layers.Embedding(
                len(self._vocab) + 1, embedding_dim
        )

    def call(self, inputs, training=True):
        embedding_id, context = inputs
        embeddings = self._embed.__call__(self._lookup.__call__(embedding_id))

        if self._context_flag:
            context_embeddings = self._context_model(context)
            # Concat along the last axis.
            # Note: this assumes a "channels last" data format!
            embeddings = tf.concat(
                    [embeddings, context_embeddings], -1
            )

        return embeddings

    def get_config(self):
        config = super().get_config()
        config.update({
            "vocab": self._vocab,
            "embedding_dim": self._embedding_dim,
            "context_model": self._context_model
        })
        return config

    @property
    def context_model(self):
        return self._context_model


@tf.keras.utils.register_keras_serializable("nets")
class HashEmbedding(BaseTFKerasModel):

    def __init__(self, hash_bins=262144, embedding_dim=32,
                 context_model=None, name="HashEmbedding"):

        super().__init__(name=name)

        self._hash_bins = hash_bins
        self._embedding_dim = embedding_dim
        self._context_model = context_model

        # Create a tf constant boolean for checking during call
        self._context_flag = self._context_model is not None

        self._lookup = tf.keras.layers.Hashing(
                num_bins=self._hash_bins
        )
        self._embed = tf.keras.layers.Embedding(
                self._hash_bins, embedding_dim
        )

    def call(self, inputs, training=True):
        embedding_id, context = inputs
        embeddings = self._embed.__call__(self._lookup.__call__(embedding_id))

        if self._context_flag:
            context_embeddings = self._context_model(context)
            # Concat along the last axis.
            # Note: this assumes a "channels last" data format!
            embeddings = tf.concat(
                    [embeddings, context_embeddings], -1
            )

        return embeddings

    def get_config(self):
        config = super().get_config()
        config.update({
            "hash_bins": self._hash_bins,
            "embedding_dim": self._embedding_dim,
            "context_model": self._context_model
        })
        return config

    @property
    def context_model(self):
        return self._context_model


@tf.keras.utils.register_keras_serializable("nets")
class DeepHashEmbedding(BaseTFKerasModel):

    _dense_config = {
        "hidden_dims": [32],
        "activation": "relu",
        "spectral_norm": True
    }

    def __init__(self, hash_bins=200000, hash_embedding_dim=64, embedding_dim=16,
                context_model=None, name="DeepHashEmbedding", **kwargs):

        super().__init__(name=name)

        self._dense_config.update(kwargs)

        self._hash_bins = hash_bins
        self._hash_embedding_dim = hash_embedding_dim
        self._embedding_dim = embedding_dim
        self._context_model = context_model

        self._embedding = HashEmbedding(
                    hash_bins=self._hash_bins,
                    embedding_dim=self._hash_embedding_dim,
                    context_model=self._context_model
        )

        self._dense_block = DenseBlock.from_config(
                self._dense_config
        )
        self._final_layer = tf.keras.layers.Dense(
                units=self._embedding_dim, activation="linear"
        )

    def call(self, inputs, training=True):

        # Embedding layer handles parsing the id and context
        raw_embeddings = self._embedding.__call__(inputs)

        embeddings = self._final_layer.__call__(
                self._dense_block.__call__(
                    raw_embeddings
                )
        )

        return embeddings

    def get_config(self):
        config = super().get_config()
        config.update({
            "hash_bins": self._hash_bins,
            "hash_embedding_dim": self._hash_embedding_dim,
            "embedding_dim": self._embedding_dim,
            "context_model": self._context_model
        })
        config.update(self._dense_config)
        return config

    @property
    def context_model(self):
        return self._context_model


@tf.keras.utils.register_keras_serializable("nets")
class SequentialDeepHashEmbeddingWithGRU(DeepHashEmbedding):
    """
    Sequential hash embeddings with multi-head self attention.
    """

    def __init__(self, hash_bins=200000, hash_embedding_dim=64,
                 embedding_dim=16, context_model=None, gru_dim=None,
                 name="SequentialDeepHashEmbeddingWithGRU", **kwargs):

        super().__init__(
            hash_bins=hash_bins,
            hash_embedding_dim=hash_embedding_dim,
            embedding_dim=embedding_dim,
            context_model=context_model,
            name=name,
            **kwargs
        )

        self._gru_dim = gru_dim
        if self._gru_dim is None:
            self._gru_dim = embedding_dim
        self._last_n_flag = self._last_n is not None

        self._gru = tf.keras.layers.GRUCell(units=self._gru_dim)

    def call(self, inputs, training=True):

        # Embedding layer handles parsing the id and context
        raw_embeddings = self._embedding.__call__(inputs)

        # If we only want to consider the last n inputs, slice.
        # Note: this assumes a "channel last" input tensor of shape
        # (batch_size, steps, embedding_dim)
        if self._last_n_flag:
            raw_embeddings = raw_embeddings[:, -self._last_n:, ...]

        # GRU cell
        gru_final_state = self._gru.__call__(raw_embeddings)

        embeddings = self._final_layer.__call__(
                self._dense_block.__call__(
                    gru_final_state
                )
        )

        return embeddings

    def get_config(self):
        config = super().get_config()
        config.update({
            "hash_bins": self._hash_bins,
            "hash_embedding_dim": self._hash_embedding_dim,
            "embedding_dim": self._embedding_dim,
            "context_model": self._context_model,
            "gru_dim": self._gru_dim,
            "last_n": self._last_n
        })
        config.update(self._dense_config)
        return config


@tf.keras.utils.register_keras_serializable("nets")
class SequentialDeepHashEmbeddingWithAttention(DeepHashEmbedding):
    """
    Sequential hash embeddings with multi-head self attention.
    """

    def __init__(self, hash_bins=200000, hash_embedding_dim=64,
                 embedding_dim=16, context_model=None, attention_key_dim=128,
                 attention_heads=4, attention_mask=False,
                 attention_pooling=False, last_n=None,
                 name="SequentialDeepHashEmbeddingWithAttention", **kwargs):

        super().__init__(
            hash_bins=hash_bins,
            hash_embedding_dim=hash_embedding_dim,
            embedding_dim=embedding_dim,
            context_model=context_model,
            name=name,
            **kwargs
        )

        self._attention_key_dim = attention_key_dim
        self._attention_heads = attention_heads
        self._attention_mask = attention_mask
        self._attention_pooling = attention_pooling
        self._last_n = last_n

        self._attention_concat = not self._attention_pooling
        self._last_n_flag = self._last_n is not None

        self._mha = MultiHeadSelfAttention(
                num_heads=self._attention_heads,
                key_dim=self._attention_key_dim,
                masking=self._attention_mask,
                pooling=self._attention_pooling,
                concat=self._attention_concat
        )

    def call(self, inputs, training=True):

        # Embedding layer handles parsing the id and context
        raw_embeddings = self._embedding.__call__(inputs)

        # If we only want to consider the last n inputs, slice.
        # Note: this assumes a "channel last" input tensor of shape
        # (batch_size, steps, embedding_dim)
        if self._last_n_flag:
            raw_embeddings = raw_embeddings[:, -self._last_n:, ...]

        # Multi-head attention cell
        raw_embeddings = self._mha.__call__(raw_embeddings)

        embeddings = self._final_layer.__call__(
                self._dense_block.__call__(
                    raw_embeddings
                )
        )

        return embeddings

    def get_config(self):
        config = super().get_config()
        config.update({
            "hash_bins": self._hash_bins,
            "hash_embedding_dim": self._hash_embedding_dim,
            "embedding_dim": self._embedding_dim,
            "context_model": self._context_model,
            "attention_key_dim": self._attention_key_dim,
            "attention_heads": self._attention_heads,
            "attention_mask": self._attention_mask,
            "attention_pooling": self._attention_pooling,
            "last_n": self._last_n
        })
        config.update(self._dense_config)
        return config

