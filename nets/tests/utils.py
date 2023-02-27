
import tensorflow as tf


class TrainSanityCallback(tf.keras.callbacks.Callback):
    def on_train_start(self, logs=None):
        self._start_loss = None
        self._min_loss = None

    def on_train_batch_end(self, batch, logs=None):
        # If it's the very first batch, assign the starting place
        batch_loss = logs["loss"]
        if batch == 0:
            self._start_loss = batch_loss
            self._min_loss = batch_loss
        else:
            self._min_loss = min(self._min_loss, batch_loss)

    def on_training_end(self, logs=None):
        assert self._min_loss >= self._start_loss, "Training did not reduce training loss."