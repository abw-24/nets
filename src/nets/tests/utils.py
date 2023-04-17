
import tensorflow as tf


class TrainSanityAssertionCallback(tf.keras.callbacks.Callback):
    """
    Simple callback to assert the batch level loss has gone down
    at least once over the course of test-training.
    """

    def on_train_start(self, logs=None):
        self._start_loss = None
        self._min_loss = None

    def on_train_batch_end(self, batch, logs=None):
        batch_loss = logs["loss"]
        # If it's the very first batch, assign the starting place
        if batch == 0:
            self._start_loss = batch_loss
            self._min_loss = batch_loss
        else:
            self._min_loss = min(self._min_loss, batch_loss)

    def on_training_end(self, logs=None):
        assert self._min_loss <= self._start_loss, \
            "Training did not reduce training loss."


def try_except_assertion_decorator(fn):
    """
    Call an instance method, assert that it worked (so that the assertion can
     be picked up by `pytest`), add exception as string message if needed.

    :param fn: Instance method
    :return: Method wrapped in try except block + an assertion
    """
    def wrapper(self):
        success = True
        msg = ""

        try:
            fn(self)
        except Exception as e:
            success = False
            msg = e

        assert success, msg

    return wrapper
