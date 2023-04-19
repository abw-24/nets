
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
    - If `last_n` is not None, only the last n attention outputs (in
    the timestep dimension) will be returned. Note: timestep dimension
    assumed to be the second dimension in the tensor (channels last).
    """

    def __init__(self, num_heads=4, key_dim=128, masking=False,
                 pooling=False, concat=False, last_n=None,
                 name="MultiHeadSelfAttention"):

        super().__init__(name=name)

        self._num_heads = num_heads
        self._key_dim = key_dim
        self._masking = masking
        self._pooling = pooling
        self._concat = concat
        self._last_n = last_n

        self._last = self._last_n is not None

        self._attention_layer = tf.keras.layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=key_dim
        )
        self._add_layer = tf.keras.layers.Add()
        self._layer_norm = tf.keras.layers.LayerNormalization()

        if self._pooling:
            self._global_pool = tf.keras.layers.GlobalAveragePooling1D()

        if self._concat:
            self._concat_layer = tf.keras.layers.Concatenate()

    def call(self, inputs):
        output = self._attention_layer.__call__(
                query=inputs,
                value=inputs,
                key=inputs,
                use_causal_mask=self._masking
        )
        embedding = self._layer_norm.__call__(
                self._add_layer.__call__([inputs, output])
        )

        if self._pooling:
            embedding = self._global_pool.__call__(embedding)
        elif self._concat:
            embedding = self._concat_layer.__call__(embedding)
        elif self._last:
            embedding = embedding[:, -self._last_n:, ...]

        return embedding

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self._num_heads,
            "key_dim": self._key_dim,
            "masking": self._masking,
            "pooling": self._pooling,
            "concat": self._concat,
            "last_n": self._n
        })
        return config