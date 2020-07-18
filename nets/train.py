
import tensorflow as tf
from nets.utils import get_tf


def model_init(model, loss, optimizer, input_dim):
    """
    Build and compile model with configured optimizer/loss.

    :param model: Instantiated keras model
    :param loss: Loss dictionary
    :param optimizer: Optimizer dictionary
    :param input_dim: Input dimension
    :return: Built and compiled nets tf.keras model
    """
    model.build(input_dim)
    model.compile(
            optimizer=get_tf(tf.keras.optimizers, optimizer),
            loss=get_tf(tf.keras.losses, loss)
    )
    return model


@tf.function
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
