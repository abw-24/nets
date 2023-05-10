
import tensorflow as tf

from nets.models.base import BaseTFKerasModel
from nets.layers.dense import DenseBlock
from nets.layers.sequence import MultiHeadSelfAttention, \
    RelativePositionEncoding, PositionEncoding
from nets.models.mixture import GatedMixture


class EmbeddingMixin(object):
    """
    Common concrete methods for StringEmbedding and HashEmbedding.
    Class attributes specified below are expected to be re-assigned in the
    init methods of inheriting children.
    """

    _embedding_dim = None
    _max_length = None
    _position_encoding = None
    _position_encoder = None
    _position_encoding_flag = None
    _embed = None
    _lookup = None
    _context_model = None
    _context_flag = None
    _masking = None

    def _set_position_encoder(self):

        self._position_encoding = self._position_encoding.lower()

        if self._position_encoding == "relative":
                self._position_encoder = RelativePositionEncoding(
                        self._embedding_dim
                )
        elif self._position_encoding == "bert":
                assert self._max_length is not None, \
                    "Requested BERT style positional encodings but did not " \
                    "provide a max sequence length"
                self._position_encoder = PositionEncoding(
                    max_length=self._max_length
                )
        else:
            raise ValueError("Requested unrecognized encoding option: {}"
                                 .format(self._position_encoding))

    def call(self, inputs, training=True):

        embedding_id, context = inputs
        embeddings = self._embed.__call__(self._lookup.__call__(embedding_id))

        # If we have a positional encoder, call and add
        if self._position_encoding_flag:
            encodings = self._position_encoder.__call__(embeddings)
            embeddings = embeddings + encodings

        # If we have context, embed and concatenate to token embeddings
        if self._context_flag:
            context_embeddings = self._context_model.__call__(context)
            # Concat along the last (embedding) axis.
            # Note: this assumes a "channels last" data format
            embeddings = tf.concat(
                    [embeddings, context_embeddings], -1
            )

        return embeddings

    def compute_mask(self, inputs, mask=None):
        """
        Unpack inputs and use the embedding layer's compute_mask to compute
        """
        if not self._masking:
            return None
        ids, context = inputs
        return self._embed.compute_mask(ids, mask=mask)

    @property
    def context_model(self):
        return self._context_model


@tf.keras.utils.register_keras_serializable("nets")
class StringEmbedding(EmbeddingMixin, BaseTFKerasModel):

    def __init__(self, vocab, embedding_dim=32, context_model=None,
                 masking=False, mask_token="[PAD]", position_encoding=None,
                 max_length=None, name="StringEmbedding"):

        super().__init__(name=name)

        self._vocab = vocab
        self._embedding_dim = embedding_dim
        self._context_model = context_model
        self._masking = masking
        self._mask_token = mask_token
        self._position_encoding = position_encoding
        self._max_length = max_length

        # Create flags to check during call
        #TODO: experiment with tf.constant vs. pure python with AutoGraph
        self._context_flag = self._context_model is not None
        self._position_encoding_flag = self._position_encoding is not None

        self._lookup = tf.keras.layers.StringLookup(
                vocabulary=self._vocab, mask_token=self._mask_token
        )
        self._embed = tf.keras.layers.Embedding(
                len(self._vocab) + 1,
                self._embedding_dim,
                mask_zero=self._masking
        )

        # If we got a position encoding arg, resolve it
        if self._position_encoding_flag:
            self._set_position_encoder()

    def get_config(self):
        config = super().get_config()
        config.update({
            "vocab": self._vocab,
            "embedding_dim": self._embedding_dim,
            "context_model": self._context_model,
            "masking": self._masking,
            "position_encoding": self._position_encoding
        })
        return config


@tf.keras.utils.register_keras_serializable("nets")
class HashEmbedding(EmbeddingMixin, BaseTFKerasModel):

    def __init__(self, hash_bins=262144, embedding_dim=32, context_model=None,
                 masking=False, position_encoding=None, max_length=None,
                 name="HashEmbedding"):

        super().__init__(name=name)

        self._hash_bins = hash_bins
        self._embedding_dim = embedding_dim
        self._context_model = context_model
        self._masking = masking
        self._position_encoding = position_encoding
        self._max_length = max_length

        # Create flags to check during call
        #TODO: experiment with tf.constant vs. pure python when using AutoGraph
        self._context_flag = self._context_model is not None
        self._position_encoding_flag = self._position_encoding is not None

        self._lookup = tf.keras.layers.Hashing(
                num_bins=self._hash_bins
        )
        self._embed = tf.keras.layers.Embedding(
                self._hash_bins, embedding_dim, mask_zero=self._masking
        )

        # If we got a position encoding, try and resolve it
        if self._position_encoding_flag:
            self._set_position_encoder()

    def get_config(self):
        config = super().get_config()
        config.update({
            "hash_bins": self._hash_bins,
            "embedding_dim": self._embedding_dim,
            "context_model": self._context_model,
            "masking": self._masking,
            "position_encoding": self._position_encoding
        })
        return config


@tf.keras.utils.register_keras_serializable("nets")
class DeepHashEmbedding(BaseTFKerasModel):
    """
    Hash embedding with additional FF layer on the end.

    Note: does not inherit from the EmbeddingMixin, instead creating a
    composite of a `HashEmbedding` submodel, followed by a dense FF block,
     and finally a last dense linear layer.
    """

    _feedforward_config = {
        "hidden_dims": [32],
        "activation": "relu",
        "spectral_norm": True
    }

    def __init__(self, hash_bins=200000, hash_embedding_dim=64, embedding_dim=16,
                context_model=None, feedforward_config=None, masking=False,
                position_encoding=None, max_length=None,
                name="DeepHashEmbedding", **kwargs):

        super().__init__(name=name, **kwargs)

        if feedforward_config is not None:
            self._feedforward_config.update(feedforward_config)

        self._hash_bins = hash_bins
        self._hash_embedding_dim = hash_embedding_dim
        self._embedding_dim = embedding_dim
        self._context_model = context_model
        self._masking = masking
        self._position_encoding = position_encoding
        self._max_length = max_length

        self._embedding = HashEmbedding(
                    hash_bins=self._hash_bins,
                    embedding_dim=self._hash_embedding_dim,
                    context_model=self._context_model,
                    masking=self._masking,
                    position_encoding=self._position_encoding,
                    max_length=self._max_length
        )

        self._dense_block = DenseBlock.from_config(
                self._feedforward_config
        )
        self._final_layer = tf.keras.layers.Dense(
                units=self._embedding_dim, activation="linear"
        )

    def call(self, inputs, training=True):

        # Embedding layer handles parsing the id and context, plus
        # any configured positional encoding
        raw_embeddings = self._embedding.__call__(inputs)

        embeddings = self._final_layer.__call__(
                self._dense_block.__call__(
                    raw_embeddings
                )
        )

        return embeddings

    def compute_mask(self, inputs, mask=None):
        """
        Use the embedding layer's compute_mask
        """
        if not self._masking:
            return None
        return self._embedding.compute_mask(inputs, mask=mask)

    def get_config(self):
        config = super().get_config()
        config.update({
            "hash_bins": self._hash_bins,
            "hash_embedding_dim": self._hash_embedding_dim,
            "embedding_dim": self._embedding_dim,
            "context_model": self._context_model,
            "masking": self._masking,
            "position_encoding": self._position_encoding
        })
        config.update(self._feedforward_config)
        return config

    @property
    def context_model(self):
        return self._context_model

    @property
    def feedforward_config(self):
        return self._feedforward_config

    @property
    def embedding(self):
        return self._embedding


@tf.keras.utils.register_keras_serializable("nets")
class SequentialDeepHashEmbeddingWithGRU(DeepHashEmbedding):
    """
    Sequential hash embeddings with GRU.
    """

    def __init__(self, hash_bins=200000, hash_embedding_dim=64,
                 embedding_dim=16, context_model=None, gru_dim=None,
                 feedforward_config=None, masking=False,
                 name="SequentialDeepHashEmbeddingWithGRU", **kwargs):

        super().__init__(
            hash_bins=hash_bins,
            hash_embedding_dim=hash_embedding_dim,
            embedding_dim=embedding_dim,
            context_model=context_model,
            feedforward_config=feedforward_config,
            masking=masking,
            position_encoding=None,
            name=name,
            **kwargs
        )

        self._gru_dim = gru_dim
        if self._gru_dim is None:
            self._gru_dim = self._embedding_dim

        self._gru = tf.keras.layers.GRU(units=self._gru_dim)

    def call(self, inputs, training=True):

        # Embedding layer handles parsing the id and context
        raw_embeddings = self._embedding.__call__(inputs)
        mask = self._embedding.compute_mask(inputs)

        # GRU
        gru_final_state = self._gru.__call__(raw_embeddings, mask=mask)

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
            "masking": self._masking
        })
        return config


@tf.keras.utils.register_keras_serializable("nets")
class SequentialDeepHashEmbeddingWithAttention(DeepHashEmbedding):
    """
    Sequential hash embeddings with multi-head self attention.
    """

    def __init__(self, hash_bins=200000, hash_embedding_dim=64,
                 embedding_dim=16, context_model=None, attention_key_dim=128,
                 attention_heads=4, attention_causal_mask=False,
                 attention_concat=False, attention_pooling=False, masking=False,
                 feedforward_config=None, position_encoding="relative", max_length=None,
                 name="SequentialDeepHashEmbeddingWithAttention", **kwargs):

        super().__init__(
            hash_bins=hash_bins,
            hash_embedding_dim=hash_embedding_dim,
            embedding_dim=embedding_dim,
            context_model=context_model,
            feedforward_config=feedforward_config,
            masking=masking,
            position_encoding=position_encoding,
            max_length=max_length,
            name=name,
            **kwargs
        )

        self._attention_key_dim = attention_key_dim
        self._attention_heads = attention_heads
        self._attention_causal_mask = attention_causal_mask
        self._attention_concat = attention_concat
        self._attention_pooling = attention_pooling

        # If either concat or pooling is `True`, make sure both are not `True`
        if self._attention_concat or self._attention_pooling:
            assert self._attention_concat != self._attention_pooling, \
                "Cannot have both 1D average pooling and concatenation. " \
                "Please set just one to `True`"

        self._mha = MultiHeadSelfAttention(
                num_heads=self._attention_heads,
                key_dim=self._attention_key_dim,
                masking=self._attention_causal_mask,
                pooling=self._attention_pooling,
                concat=self._attention_concat
        )

    def call(self, inputs, training=True):

        # Embedding layer handles parsing the id and context
        raw_embeddings = self._embedding.__call__(inputs)
        mask = self._embedding.compute_mask(inputs)

        # Propagate masks. MHA layer will reformat as needed
        raw_embeddings = self._mha.__call__(raw_embeddings, mask=mask)

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
            "attention_causal_mask": self._attention_causal_mask,
            "attention_pooling": self._attention_pooling,
            "attention_concat": self._attention_concat,
            "masking": self._masking,
            "feedforward_config": self._feedforward_config,
            "position_encoding": self._position_encoding
        })
        config.update(self._feedforward_config)
        return config


@tf.keras.utils.register_keras_serializable("nets")
class SequentialDeepHashEmbeddingMixtureOfExperts(GatedMixture):
    """
    Sequential mixture of experts.

    - Long range model is hash embeddings + multi-head self attention + FF
    - Short range model is hash embeddings + GRU + FF
    """

    def __init__(self, hash_embedding_dim=128, embedding_dim=32, masking=False,
                 context=False, name="SequentialDeepHashEmbeddingMixture",
                 **kwargs):

        super().__init__(
                n_experts=2,
                expert_dim=embedding_dim,
                name=name,
                **kwargs
        )

        self._hash_embedding_dim = hash_embedding_dim
        self._embedding_dim = embedding_dim
        self._masking = masking
        self._context = context

        long_context_model = None
        short_context_model = None

        # If context is specified, create simple FF layers for each expert
        if self._context:
            long_context_model = tf.keras.layers.Dense(
                    units=self._embedding_dim, activation="relu"
            )
            short_context_model = tf.keras.layers.Dense(
                    units=self._embedding_dim, activation="relu"
            )

        # Long range model is multi-head self attention + FF
        long_range_model = SequentialDeepHashEmbeddingWithAttention(
                hash_embedding_dim=self._hash_embedding_dim,
                embedding_dim=self._embedding_dim,
                feedforward_config={"hidden_dims": [64]},
                position_encoding="relative",
                attention_key_dim=128,
                attention_heads=4,
                attention_concat=True,
                attention_causal_mask=False,
                masking=self._masking,
                context_model=long_context_model
        )

        # Short range model is GRU + FF
        short_range_model = SequentialDeepHashEmbeddingWithGRU(
                hash_embedding_dim=self._hash_embedding_dim,
                embedding_dim=self._embedding_dim,
                feedforward_config={"hidden_dims": [64]},
                masking=self._masking,
                context_model=short_context_model
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
            "embedding_dim": self._embedding_dim,
            "masking": self._masking,
            "context": self._context
        })
        return config
