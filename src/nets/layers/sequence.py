
import tensorflow as tf


@tf.keras.utils.register_keras_serializable("nets")
class MultiHeadSelfAttention(tf.keras.layers.Layer):
    """
    Self-attention with residual connections, as in Transformer setups.

    - If `masking` is set to True, the attention layer is called with the
    `use_causal_mask` arg set to True.
    - If `pooling` is set to True, the MHA output is averaged over the
    timestep dimension.
    - If `concat` is set to True, the MHA timesteps are concatenated.
    """

    def __init__(self, num_heads=4, key_dim=128, masking=False,
                 pooling=False, concat=False, name="MultiHeadSelfAttention"):

        super().__init__(name=name)

        self._num_heads = num_heads
        self._key_dim = key_dim
        self._masking = masking
        self._pooling = pooling
        self._concat = concat

        self._attention_layer = tf.keras.layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=key_dim
        )
        self._add_layer = tf.keras.layers.Add()
        self._layer_norm = tf.keras.layers.LayerNormalization()

        if self._pooling:
            self._global_pool = tf.keras.layers.GlobalAveragePooling1D()

        if self._concat:
            self._flatten_layer = tf.keras.layers.Flatten()

    def call(self, inputs, mask=None, training=True):
        """
        Apply the attention layer and the residual connection, then either
        pooling or flattening if configured, otherwise return all timesteps
         (same shape as input).

         Mask handling is needed to expand (batch, seq_len) masks to
         (batch, seq_len, seq_len) masks for self-attention
        """
        # Mask reshape
        if mask is not None:
            mask = tf.repeat(
                    tf.expand_dims(mask, axis=1),
                    [inputs.shape[1]],
                    axis=1
            )
        # Pass inputs and mask to attention layer
        output = self._attention_layer.__call__(
                query=inputs,
                value=inputs,
                key=inputs,
                use_causal_mask=self._masking,
                attention_mask=mask
        )
        # Add and layer norm (mask preserving)
        embedding = self._layer_norm.__call__(
                self._add_layer.__call__(
                        [inputs, output]
                )
        )

        # If either of these ops are performed, masks are destroyed
        if self._pooling:
            embedding = self._global_pool.__call__(embedding)
        elif self._concat:
            embedding = self._flatten_layer.__call__(embedding)

        return embedding

    def compute_mask(self, inputs, mask=None):
        """
        If either pooling or concat is set to `True`, masks are destroyed.
        Otherwise, they're returned unchaged.
        """
        if not self._masking:
            return None
        if self._pooling:
            return None
        if self._concat:
            return None

        return mask

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self._num_heads,
            "key_dim": self._key_dim,
            "masking": self._masking,
            "pooling": self._pooling,
            "concat": self._concat
        })
        return config


@tf.keras.utils.register_keras_serializable("nets")
class PositionEncoding(tf.keras.layers.Layer):
    """
    BERT-style positional encoding. Requires a maximum length.

    Based on the `tfm` implementation.
    """

    def __init__(self, max_length, seq_axis=1, **kwargs):

        super().__init__(**kwargs)

        # Make sure we set `supports_masking` to True to pass through
        # masks coming from embedded inputs
        self._supports_masking = True

        self._max_length = max_length
        self._initializer = tf.keras.initializers.get("glorot_uniform")
        self._seq_axis = seq_axis

        self._position_embeddings = None

    def build(self, input_shape):

        embedded_dim = input_shape.as_list()[-1]
        self._position_embeddings = self.add_weight(
                "embeddings",
                shape=[self._max_length, embedded_dim],
                initializer=self._initializer
        )

        super().build(input_shape)

    def call(self, inputs):
        """
        Grab embeddings from the lookup for indices based on the shape of the
        input (the size of the `seq_dim`)
        """
        input_shape = tf.shape(inputs)

        # Grab encodings for current sequence positions (0 through seq_len)
        seq_len = input_shape[self._seq_axis]
        position_embeddings = self._position_embeddings[:seq_len, :]

        # Expand dims and broadcast to full (batch, seq_len, embedding_dim)
        position_embeddings = tf.expand_dims(position_embeddings, axis=0)

        return tf.broadcast_to(position_embeddings, input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            "max_length": self._max_length,
            "seq_axis": self._seq_axis,
        })
        return config


@tf.keras.utils.register_keras_serializable("nets")
class RelativePositionEncoding(tf.keras.layers.Layer):
    """
    Attention-is-all-you-need style positional encodings using trigonometric
      functions. Does not require a maxium length.

      Based on the `tfm` implementation.
    """

    def __init__(self, hidden_dim, min_timescale=1.0,
                 max_timescale=1.0e4, **kwargs):

        # Unless otherwise specified, default to float32
        if "dtype" not in kwargs:
            kwargs["dtype"] = "float32"

        super().__init__(**kwargs)

        # Make sure we set `supports_masking` to True to pass through
        # masks coming from embedded inputs
        self._supports_masking = True

        self._hidden_dim = hidden_dim
        self._min_timescale = min_timescale
        self._max_timescale = max_timescale

    def call(self, inputs, training=True):

        input_shape = tf.shape(inputs)
        length = input_shape[1]
        position = tf.cast(tf.range(length), tf.float32)
        num_timescales = self._hidden_dim // 2

        min_timescale, max_timescale = self._min_timescale, self._max_timescale

        log_timescale_increment = (
            tf.math.log(float(max_timescale) / float(min_timescale)) /
            (tf.cast(num_timescales, tf.float32) - 1)
        )
        inv_timescales = min_timescale * tf.exp(
                tf.cast(tf.range(num_timescales), tf.float32) *
                -log_timescale_increment
        )
        scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(
                inv_timescales, 0
        )
        position_embeddings = tf.concat(
                [tf.sin(scaled_time), tf.cos(scaled_time)], 1
        )

        # Expand dims and broadcast position embeddings back to input_shape
        position_embeddings = tf.expand_dims(position_embeddings, axis=0)

        return tf.broadcast_to(position_embeddings, input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_dim": self._hidden_dim,
            "min_timescale": self._min_timescale,
            "max_timescale": self._max_timescale,
        })
        return config
