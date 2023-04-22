# nets
Tensorflow 2.x network architectures

* Focus is on neural recommenders (see `nets.models.recommeder`)
* Models are subclassed from `tf.keras.Model` and `tfrs.models.Model` to retain the high-level `keras` APIs
* Tensor computations are housed in custom layers and/or submodels, making for easy saving and reuse
* Test suite can be run with `pytest` as the test runner
    * Note: Integration tests do not mock training or saving, so the full suite may take a few minutes to run

### Example usage: Sequential item-to-item
```
import nets.models.recommender as rec
import nets.models.mlp as mlp

import tensorflow as tf

# To create custom models, build query and candidate models of your choosing 
# and pass as constructor arguments to the appropriate model class (in this
# case TwoTowerRetrieval). After defining, compile and use like any
# other keras model

query_model = rec.embedding.SequentialDeepHashEmbeddingWithAttention(
        hash_embedding_dim=128,
        embedding_dim=32,
        hidden_dims=[64],
        attention_key_dim=256,
        attention_heads=4,
        attention_concat=True,
        attention_masking=False
)
candidate_model = rec.embedding.DeepHashEmbedding(
    hash_embedding_dim=128,
    embedding_dim=32,
    hidden_dims=[64],
    activation="relu"
)
model = rec.retrieval.TwoTowerRetrieval(
    query_model=query_model,
    candidate_model=candidate_model,
    query_id="context_movie_titles",
    candidate_id="movie_title",
)
model.compile(
    optimizer=tf.keras.optizers.Adam(learning_rate=0.001)
)

# You can also use models with pre-defined architectures like the
# sequential MoE retrieval model, which has a query model that mixes a
# multi-head self attention model for long range dependencies and a GRU
# model for short range dependencies

model = rec.retrieval.SequentialMixtureOfExpertsRetrieval(
    query_id="context_movie_titles",
    candidate_id="movie_title",
)
# Compile and fit as you would any other keras model
model.compile(
    optimizer=tf.keras.optizers.Adam(learning_rate=0.001)
)
```
