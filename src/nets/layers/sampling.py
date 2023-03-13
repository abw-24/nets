
import tensorflow as tf


@tf.keras.utils.register_keras_serializable("nets")
class GaussianSampling(tf.keras.layers.Layer):
    """
    Samples from distribution defined by the latent layer values to
    generate values from which to decode.
    """

    def call(self, inputs):
        """
        Inputs expected to be
        :param inputs:
        :return:
        """
        z_mean, z_log_var = inputs
        epsilon = tf.keras.backend.random_normal(
                shape=(tf.shape(z_mean)[0],
                       tf.shape(z_mean)[1])
        )
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def get_config(self):
        return super(GaussianSampling, self).get_config()



