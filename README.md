# nets
Tensorflow 2.x network architectures

* Focus is on neural recommenders (see `nets.models.recommeder`)
* Models are subclassed from `tf.keras.Model` and `tfrs.models.Model` to retain the high-level `keras` APIs
* Tensor computations are housed in pre-registered custom layers and/or submodels, making for easy saving and reuse
* Test suite can be run with `pytest` as the test runner
    * Note: Integration tests do not mock training or saving, so the full suite may take a few minutes to run

### Example usage: Sequential item-to-item
```python
import nets.models.recommender.retrieval as ret
import nets.models.recommender.embedding as emb
import nets.models.mlp as mlp

import tensorflow as tf

# To create custom models, build query and candidate models of your choosing 
# and pass as constructor arguments to the appropriate model class (in this
# case TwoTowerRetrieval). After defining, compile and use like any
# other keras model

query_model = emb.SequentialDeepHashEmbeddingWithAttention(
        hash_embedding_dim=128,
        embedding_dim=32,
        hidden_dims=[64],
        attention_key_dim=256,
        attention_heads=4,
        attention_concat=True,
        attention_masking=False
)
candidate_model = emb.DeepHashEmbedding(
    hash_embedding_dim=128,
    embedding_dim=32,
    hidden_dims=[64],
    activation="relu"
)
model = ret.TwoTowerRetrieval(
    query_model=query_model,
    candidate_model=candidate_model,
    query_id="context_movie_titles",
    candidate_id="movie_title",
)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
)

# You can also use models with pre-defined architectures like this
# sequential mixture-of-experts retrieval model, which has a query model
# that mixes multi-head self attention for long range dependencies and
# GRU for short range dependencies on item interaction history

model = ret.SequentialMixtureOfExpertsRetrieval(
    query_id="context_movie_titles",
    candidate_id="movie_title",
)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
)
```
