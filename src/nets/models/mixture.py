
import tensorflow as tf
import tensorflow_addons as tfa

from nets.models.base import BaseTFKerasModel


@tf.keras.utils.register_keras_serializable("nets")
class GatedMixtureABC(BaseTFKerasModel):
    """
    Gated mixture of experts.

    Takes raw inputs, passes them through each expert and gate
    layer pair, elementwise multiplies the pairs, and sums,
    returning a tensor of shape (batch_size, embedding_dim).

    Inheritors should define the expert models in the init method and
     assign a list of the experts to a `self._experts` instance variable.
    """

    def __init__(self, n_experts, expert_dim, spectral_norm=False,
                 name="GatedMixtureABC", **kwargs):

        super().__init__(name=name, **kwargs)

        self._n_experts = n_experts
        self._expert_dim = expert_dim
        self._spectral_norm = spectral_norm

        self._gate_layers = []

        for i in range(self._n_experts):
            dense_layer = tf.keras.layers.Dense(
                    units=expert_dim, activation="sigmoid"
            )
            if self._spectral_norm:
                dense_layer = tfa.layers.SpectralNormalization(
                            dense_layer
                )
            self._gate_layers.append(dense_layer)

    def call(self, inputs, training=True):

        outputs = None

        # Iterate over experts and gate layers, take the elementwise product
        # and add to the outputs sum vector.
        # AutoGraph convertible -- no side effects
        for expert, gate_layer in zip(self._experts, self._gate_layers):
            expert_output = expert.__call__(inputs)
            gates = gate_layer.__call__(inputs)
            gated_expert = tf.math.multiply(gates, expert_output)
            if outputs is None:
                outputs = gated_expert
            else:
                outputs = tf.math.add(outputs, gated_expert)

        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({
            "n_experts": self._n_experts,
            "expert_dim": self._expert_dim,
            "spectral_norm": self._spectral_norm
        })
        return config

    @property
    def experts(self):
        return self._experts