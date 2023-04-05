# nets
Tensorflow 2.x network architectures

* Focus is on neural recommenders (see `nets.models.recommeder` modules) and representation learning (see `nets.models.vae`)
* Models are subclassed from `tf.keras.Model` and `tfrs.models.Model` to retain the high-level `keras` APIs
* Tensor computations are primarily housed in custom block layers and/or submodels, making for easy saving and reuse
* Integration and unit tests can be run manually with the `pytest` test runner
    * Model integration tests (which include fitting and saving to disk) are not mocked, and as a result typically take a bit of time, on the order of minutes
