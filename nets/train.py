
import tensorflow as tf
from nets.utils import get_tf


def model_init(model, config, input_dim):
    """
    Build and compile model with configured optimizer/loss.
    :param mode_class: TF keras (nets) model class
    :param config: Configuration dictionary
    :return: Built and compiled nets tf.keras model
    """
    model.build(input_dim)
    model.compile(
            optimizer=get_tf(tf.keras.optimizers, config["optimizer"]),
            loss=get_tf(tf.keras.losses, config["loss"])
    )
    return model


def grad(model, x, y):
    """
    Takes a compiled model and takes the gradients wrt
    the loss over the given inputs and targets
    :param model: Compiled tf.keras.Model / nets model
    :param x: Training batch
    :param y: Training batch
    :return: Loss, gradients
    """
    with tf.GradientTape() as tape:
        prediction = model(x, training=True)
        loss_value = model.loss(y, prediction)
        grads = tape.gradient(loss_value, model.trainable_variables)

    return loss_value, grads