
import tensorflow as tf


@tf.keras.utils.register_keras_serializable("nets")
class MultiHeadSelfAttention(tf.keras.layers.Layer):
    """
    Self-attention with residual connections, as in Transformer setups.
    Optionally can set `masking` to True for "causal" training with a
    1-step shifted target vector, as in the original paper.
    """

    def __init__(self, num_heads=4, key_dim=128, masking=False,
                 name="MultiHeadSelfAttention"):

        super().__init__(name=name)

        self._num_heads = num_heads
        self._key_dim = key_dim
        self._masking = masking

        self._attention = tf.keras.layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=key_dim
        )
        self._add = tf.keras.layers.Add()
        self._layer_norm = tf.keras.layers.LayerNormalization()

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs):
        output = self._attention(
                query=inputs,
                value=inputs,
                key=inputs,
                use_causal_mask=self._masking
        )
        embedding = self._add([inputs, output])
        embedding = self._layer_norm(embedding)
        return embedding

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self._num_heads,
            "key_dim": self._key_dim,
            "masking": self._masking
        })
        return config