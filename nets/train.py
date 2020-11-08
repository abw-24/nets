
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

    if loss is not None:
        if isinstance(loss, dict):
            loss_obj = get_tf(tf.keras.losses, loss)
        elif isinstance(loss, tf.keras.losses.Loss):
            loss_obj = loss
        else:
            raise ValueError("Loss must be either a dictionary or tf loss instance.")
    else:
        loss_obj = loss

    if optimizer is not None:
        if isinstance(optimizer, dict):
            optimizer_obj = get_tf(tf.keras.optimizers, optimizer)
        elif isinstance(optimizer, tf.keras.optimizers.Optimizer):
            optimizer_obj = optimizer
        else:
            raise ValueError("Optimizer must be either a dictionary or tf optimizer instance.")
    else:
        return None

    model.build(input_dim)
    model.compile(
            optimizer=optimizer_obj,
            loss=loss_obj
    )

    return model


#TODO: specify a generic signature to cut down on retracing
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
