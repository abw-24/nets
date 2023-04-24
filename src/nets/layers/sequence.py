
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