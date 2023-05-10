
import tensorflow as tf


@tf.keras.utils.register_keras_serializable("nets")
class GaussianSampling(tf.keras.layers.Layer):
    """
    Samples from a gaussian random vector defined by the inputs.
    """

    def call(self, inputs):

        z_mean, z_log_var = inputs
        epsilon = tf.keras.backend.random_normal(
                shape=(tf.shape(z_mean)[0],
                       tf.shape(z_mean)[1])
        )
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
