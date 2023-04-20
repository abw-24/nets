# nets
Tensorflow 2.x network architectures

* Focus is on neural recommenders (see `nets.models.recommeder` modules)
* Models are subclassed from `tf.keras.Model` and `tfrs.models.Model` to retain the high-level `keras` APIs
* Tensor computations are primarily housed in custom block layers and/or submodels, making for easy saving and reuse
* Test suite can be run with `pytest` as the test runner
    * Note: Integration tests do not mock training or saving, so the full suite may take a few minutes to run

### Example usage: Sequential item-item
```
import nets.models.recommender as rec
import nets.models.mlp as mlp

import tensorflow as tf

# Create a sequential MoE retrieval model
model = rec.retrieval.SequentialMixtureOfExpertsRetrieval(
    query_id="context_movie_titles",
    candidate_id="movie_title",
)
# Compile and fit as you would any other keras model
model.compile(
    optimizer=tf.keras.optizers.Adam(learning_rate=0.001)
)
```
