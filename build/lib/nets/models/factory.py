
"""
Model class factories.
"""

from nets.models.mlp import MLP
from nets.models.vae import VAE


class MLPFactory(object):

    @classmethod
    def apply(cls, config):

        input_shape = config.get("input_shape", None)
        hidden_dims = config.get("hidden_dims")
        output_dim = config.get("output_dim")
        activation = config.get("activation", "relu")
        output_activation = config.get("output_activation", None)
        k_regularizer = config.get("kernel_regularizer", None)
        a_regularizer = config.get("activity_regularizer", None)

        return MLP(
                hidden_dims=hidden_dims,
                output_dim=output_dim,
                input_shape=input_shape,
                activation=activation,
                output_activation=output_activation,
                kernel_regularizer=k_regularizer,
                activity_regularizer=a_regularizer
        )


class VAEFactory(object):

    @classmethod
    def apply(cls, config):

        encoding_dims = config.get("encoding_dims")
        latent_dim = config.get("latent_dim")
        input_shape = config.get("input_shape", None)
        activation = config.get("activation", "relu")
        activity_regularizer = config.get("activity_regularizer", None)
        reconstruction_activation = config.get("reconstruction_activation", None)
        sparse_flag = config.get("sparse_flag", False)

        return VAE(
            encoding_dims=encoding_dims,
            latent_dim=latent_dim,
            input_shape=input_shape,
            activation=activation,
            activity_regularizer=activity_regularizer,
            reconstruction_activation=reconstruction_activation,
            sparse_flag=sparse_flag
        )