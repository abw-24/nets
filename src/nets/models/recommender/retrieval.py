
import tensorflow as tf
import tensorflow_recommenders as tfrs

from .base import TwoTowerABC, TwoTowerTrait
from nets.models.recommender.embedding import HashEmbedding, DeepHashEmbedding
from nets.layers.dense import GatedMixture


@tf.keras.utils.register_keras_serializable("nets")
class TwoTowerRetrieval(TwoTowerTrait, TwoTowerABC):
    """
    """
    def __init__(self, query_model, candidate_model, query_id,
                 candidate_id, query_context_features=None,
                 candidate_context_features=None, name="TwoTowerRetrieval"):

        super().__init__(name=name)

        self._task = tfrs.tasks.Retrieval()

        self._query_model = query_model
        self._candidate_model = candidate_model
        self._query_id = query_id
        self._candidate_id = candidate_id
        self._query_context_features = query_context_features
        self._candidate_context_features = candidate_context_features

        self._query_context_flag = \
            self._query_context_features is not None
        self._candidate_context_flag = \
            self._candidate_context_features is not None

    def call(self, inputs, training=False):
        """
        """
        query_embeddings = self._query_model_with_context(inputs)
        candidate_embeddings = self._candidate_model_with_context(inputs)

        return query_embeddings, candidate_embeddings

    @tf.function
    def compute_loss(self, features, training=True):
        """
        Compute loss for a batch by invoking the task
        """
        query_embeddings, candidate_embeddings = self.__call__(features)
        return self._task.__call__(
                query_embeddings=query_embeddings,
                candidate_embeddings=candidate_embeddings,
                compute_metrics=not training,
                compute_batch_metrics=not training
        )


class MixtureOfAttentionExpertsRetrieval(TwoTowerRetrieval):

    """
    Sequential mixture of experts (MoE). Experts are two multi-head attention
    networks with
    """

    def __init__(self, query_id, candidate_id, share_embeddings=True,
                 embedding_dim=32, attention_pooling=True,
                 query_context_features=None, candidate_context_features=None,
                 name="SelfAttentionMixtureOfExpertsRetrieval"):

        self._share_embeddings = share_embeddings
        self._embedding_dim = embedding_dim
        self._attention_pooling = attention_pooling
        self._hash_embeddings_dim = 128

        # If the experts share embeddings, create them
        shared_embeddings = None
        if self._share_embeddings:
            shared_embeddings = HashEmbedding(
                embedding_dim=self._hash_embeddings_dim
            )

        # Long range model has more heads, larger attention dim
        long_range_model = DeepHashEmbedding(
                embeddings=shared_embeddings,
                hash_embedding_dim=self._hash_embeddings_dim,
                embedding_dim=self._embedding_dim,
                hidden_dims=[64],
                attention_key_dim=128,
                attention_heads=4,
                attention_pooling=self._attention_pooling
        )

        # Short range model with just 1 head, small attention dim
        short_range_model = DeepHashEmbedding(
                embeddings=shared_embeddings,
                hash_embedding_dim=self._hash_embeddings_dim,
                embedding_dim=self._embedding_dim,
                hidden_dims=[64],
                attention_key_dim=64,
                attention_heads=1,
                attention_pooling=self._attention_pooling
        )

        query_model = GatedMixture(
            experts=[long_range_model, short_range_model],
            expert_dim=self._embedding_dim
        )

        candidate_model = DeepHashEmbedding(
                hash_embedding_dim=self._hash_embeddings_dim,
                hidden_dims=[64],
                embedding_dim=self._embedding_dim
        )

        super().__init__(
                query_model=query_model,
                candidate_model=candidate_model,
                query_id=query_id,
                candidate_id=candidate_id,
                query_context_features=query_context_features,
                candidate_context_features=candidate_context_features,
                name=name
        )

    def build(self, input_shape):
        # Make sure to build the gated mixture of experts model
        self._query_model.build(input_shape=input_shape)
        super().build(input_shape=input_shape)