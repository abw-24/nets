
import tensorflow as tf
import tensorflow_addons as tfa


@tf.keras.utils.register_keras_serializable("nets")
class DenseBlock(tf.keras.layers.Layer):

    def __init__(self, hidden_dims, activation="relu", kernel_regularizer=None,
                 activity_regularizer=None, spectral_norm=False, **kwargs):

        super(DenseBlock, self).__init__(**kwargs)

        if isinstance(hidden_dims, int):
            self._hidden_dims = [hidden_dims]
        else:
            self._hidden_dims = hidden_dims
        if isinstance(activation, str):
            self._activation = [activation]*len(self._hidden_dims)
        else:
            self._activation = activation

        self._kernel_regularizer = kernel_regularizer
        self._activity_regularizer = activity_regularizer
        self._spectral_norm = spectral_norm

        self._block_layers = []
        for d, a in zip(self._hidden_dims, self._activation):
            dense = tf.keras.layers.Dense(
                    units=d,
                    activation=a,
                    kernel_regularizer=self._kernel_regularizer,
                    activity_regularizer=self._activity_regularizer
            )

            if self._spectral_norm:
                self._block_layers.append(tfa.layers.SpectralNormalization(layer=dense))
            else:
                self._block_layers.append(dense)

    def call(self, inputs, training=False):
        outputs = inputs
        # Should be AutoGraph convertible -- no side effects
        for layer in self._block_layers:
            outputs = layer.__call__(outputs)
        return outputs

    def get_config(self):
        config = super(DenseBlock, self).get_config()
        config.update({
            "hidden_dims": self._hidden_dims,
            "activation": self._activation,
            "activity_regularizer": self._activity_regularizer,
            "kernel_regularizer": self._kernel_regularizer,
            "spectral_norm": self._spectral_norm
        })
        return config


@tf.keras.utils.register_keras_serializable("nets")
class GatedMixture(tf.keras.layers.Layer):
    """
    Gated mixture of experts.

    Takes raw inputs, passes them through each expert and gate
    layer pair, elementwise multiplies the pairs, and sums,
    returning a tensor of shape (batch_size, embedding_dim)
    """

    def __init__(self, experts, embedding_dim, spectral_norm=False,
                 name="GatedMixture", **kwargs):

        super().__init__(**kwargs)

        self._experts = experts
        self._n_experts = len(experts)
        self._expert_dim = embedding_dim
        self._spectral_norm = spectral_norm
        self._batch_size = None

        self._gate_layers = []

        for i in range(self._n_experts):
            dense_layer = tf.keras.layers.Dense(
                    units=embedding_dim, activaton="sigmoid"
            )
            if self._spectral_norm:
                dense_layer = tfa.layers.SpectralNormalization(
                            dense_layer
                )
            self._gate_layers.append(dense_layer)

        self._add = tf.keras.layers.Add()

    def build(self, input_shape):
        self._batch_size = input_shape[0]
        super().build(input_shape)

    def call(self, inputs, training=True):

        # Create an empty tensor to add to
        outputs = tf.zeros(shape=(self._batch_size, self._expert_dim))

        # Iterate over experts and gate layers, take the elementwise product
        # and add to the outputs sum vector.
        # Should be AutoGraph convertible -- no side effects
        for expert, gate_layer in zip(self._experts, self._gate_layers):
            expert_output = expert.__call__(inputs)
            gates = gate_layer.__call__(inputs)
            gated_expert = tf.math.multiply(gates, expert_output)
            outputs = self._add.__call__([outputs, gated_expert])

        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({
            "experts": self._experts,
            "embedding_dim": self._embedding_dim
        })
        return config