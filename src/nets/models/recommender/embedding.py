
import tensorflow as tf

from nets.models.base import BaseTFKerasModel
from nets.layers.dense import DenseBlock
from nets.layers.sequence import MultiHeadSelfAttention
from nets.models.mixture import GatedMixtureABC


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

    _feedforward_config = {
        "hidden_dims": [32],
        "activation": "relu",
        "spectral_norm": True
    }

    def __init__(self, hash_bins=200000, hash_embedding_dim=64, embedding_dim=16,
                context_model=None, name="DeepHashEmbedding",
                 feedforward_config=None, **kwargs):

        super().__init__(name=name)

        if feedforward_config is not None:
            self._feedforward_config.update(feedforward_config)

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
                self._feedforward_config
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
        config.update(self._feedforward_config)
        return config

    @property
    def context_model(self):
        return self._context_model

    @property
    def feedforward_config(self):
        return self._feedforward_config


@tf.keras.utils.register_keras_serializable("nets")
class SequentialDeepHashEmbeddingWithGRU(DeepHashEmbedding):
    """
    Sequential hash embeddings with GRU
    """

    def __init__(self, hash_bins=200000, hash_embedding_dim=64,
                 embedding_dim=16, context_model=None, gru_dim=None,
                 last_n=None, name="SequentialDeepHashEmbeddingWithGRU",
                 feedforward_config=None, **kwargs):

        super().__init__(
            hash_bins=hash_bins,
            hash_embedding_dim=hash_embedding_dim,
            embedding_dim=embedding_dim,
            context_model=context_model,
            feedforward_config=feedforward_config,
            name=name,
            **kwargs
        )

        self._gru_dim = gru_dim
        if self._gru_dim is None:
            self._gru_dim = self._embedding_dim

        self._last_n = last_n
        self._last_n_flag = self._last_n is not None

        self._gru = tf.keras.layers.GRU(units=self._gru_dim)

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
            "feedforward_config": self._feedforward_config,
            "gru_dim": self._gru_dim,
            "last_n": self._last_n
        })
        return config


@tf.keras.utils.register_keras_serializable("nets")
class SequentialDeepHashEmbeddingWithAttention(DeepHashEmbedding):
    """
    Sequential hash embeddings with multi-head self attention.
    """

    def __init__(self, hash_bins=200000, hash_embedding_dim=64,
                 embedding_dim=16, context_model=None, attention_key_dim=128,
                 attention_heads=4, attention_mask=False, attention_concat=False,
                 attention_pooling=False, last_n=None, feedforward_config=None,
                 name="SequentialDeepHashEmbeddingWithAttention", **kwargs):

        super().__init__(
            hash_bins=hash_bins,
            hash_embedding_dim=hash_embedding_dim,
            embedding_dim=embedding_dim,
            context_model=context_model,
            feedforward_config=feedforward_config,
            name=name,
            **kwargs
        )

        self._attention_key_dim = attention_key_dim
        self._attention_heads = attention_heads
        self._attention_mask = attention_mask
        self._attention_concat = attention_concat
        self._attention_pooling = attention_pooling
        self._last_n = last_n

        # If either concat or pooling is `True`, make sure both are not `True`
        if self._attention_concat or self._attention_pooling:
            assert self._attention_concat != self._attention_pooling, \
                "Cannot have both 1D average pooling and concatenation. " \
                "Please set just one to `True`"

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
            "attention_concat": self._attention_concat,
            "last_n": self._last_n
        })
        config.update(self._feedforward_config)
        return config


@tf.keras.utils.register_keras_serializable("nets")
class SequentialDeepHashEmbeddingMixtureOfExperts(GatedMixtureABC):
    """
    Sequential mixture of experts.

    - Long range model is hash embeddings + multi-head self attention + FF
    - Short range model is hash embeddings + GRU + FF
    """

    def __init__(self, hash_embedding_dim=128, embedding_dim=32,
                 name="SequentialDeepHashEmbeddingMixture", **kwargs):

        super().__init__(
                n_experts=2,
                expert_dim=embedding_dim,
                name=name,
                **kwargs
        )

        self._hash_embedding_dim = hash_embedding_dim
        self._embedding_dim = embedding_dim

        # Long range model is multi-head self attention + FF
        long_range_model = SequentialDeepHashEmbeddingWithAttention(
                hash_embedding_dim=self._hash_embedding_dim,
                embedding_dim=self._embedding_dim,
                feedforward_config={"hidden_dims": [64]},
                attention_key_dim=128,
                attention_heads=4,
                attention_concat=True
        )

        # Short range model is GRU + FF
        short_range_model = SequentialDeepHashEmbeddingWithGRU(
                hash_embedding_dim=self._hash_embedding_dim,
                embedding_dim=self._embedding_dim,
                feedforward_config={"hidden_dims": [64]}
        )

        # Set experts
        self._experts = [short_range_model, long_range_model]

    def call(self, inputs, training=True):
        """
        Override base GatedMixture to use expert output as the gate input
        """
        outputs = None

        # Iterate over experts and gate layers, take the elementwise product
        # and add to the outputs sum vector. In this version, we use the
        # expert output vector as the input to the gating layer.
        # AutoGraph convertible -- no side effects
        for expert, gate_layer in zip(self._experts, self._gate_layers):
            expert_output = expert.__call__(inputs)
            gates = gate_layer.__call__(expert_output)
            gated_expert = tf.math.multiply(gates, expert_output)
            if outputs is None:
                outputs = gated_expert
            else:
                outputs = tf.math.add(outputs, gated_expert)

        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({
            "hash_embedding_dim": self._hash_embedding_dim,
            "embedding_dim": self._embedding_dim
        })
        return config
