# nets
Tensorflow 2.x network architectures

* Focus is on neural recommenders (see `nets.models.recommeder` modules) and representation learning (see `nets.models.vae`)
* Models are subclassed from `tf.keras.Model` and `tfrs.models.Model` to retain the high-level `keras` APIs
* Tensor computations are primarily housed in custom block layers and/or submodels, making for easy saving and reuse
* Test suite can be run with `pytest` as the test runner
    * Note: Integration tests do not mock training or saving, so the full suite may take a few minutes to run

### Example usage: User-item Two Tower Retrieval + Ranking
```
import nets.models.recommender as rec
import nets.models.mlp as mlp

import tensorflow as tf


# Create hash embeddings with dense layers on top for user/item token embeddings
query_model = rec.embedding.DeepHashEmbedding(
    hash_embedding_dim=32,
    embedding_dim=16,
    hidden_dims=[24],
    activation="relu"
)
candidate_model = rec.embedding.DeepHashEmbedding(
    hash_embedding_dim=64,
    embedding_dim=16,
    hidden_dims=[32],
    activation="relu"
)

# Create a dense feedforward network to predict ratings for ranking
target_model = mlp.MLP(
    hidden_dims=[8],
    output_dim=1,
    activation="relu",
    output_activation="linear",
    spectral_norm=True
)
# Create the multitask model, passing the submodels and string names for
# the various fields to the constructor
model = rec.multitask.TwoTowerMultiTask(
    target_model=target_model,
    query_model=query_model,
    candidate_model=candidate_model,
    query_id="user_id",
    candidate_id="movie_title",
    rank_target="user_ratings",
    balance=0.5
)
# Compile
model.compile(
    optimizer=tf.keras.optizers.Adam(learning_rate=0.001)
)
```
