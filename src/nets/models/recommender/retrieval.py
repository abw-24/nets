
import tensorflow as tf
import tensorflow_recommenders as tfrs

from .base import TwoTowerABC, TwoTowerMixin
from nets.models.recommender.embedding import DeepHashEmbedding, \
    SequentialDeepHashEmbeddingMixtureOfExperts


@tf.keras.utils.register_keras_serializable("nets")
class TwoTowerRetrieval(TwoTowerMixin, TwoTowerABC):
    """
    Basic two tower retrieval.
    """
    def __init__(self, query_model, candidate_model, query_id,
                 candidate_id, query_context_features=None,
                 candidate_context_features=None, name="TwoTowerRetrieval",
                 **kwargs):

        super().__init__(name=name, **kwargs)

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
        query_embeddings, candidate_embeddings = self.__call__(features)
        return self._task.__call__(
                query_embeddings=query_embeddings,
                candidate_embeddings=candidate_embeddings,
                compute_metrics=not training,
                compute_batch_metrics=not training
        )


@tf.keras.utils.register_keras_serializable("nets")
class SequentialMixtureOfExpertsRetrieval(TwoTowerRetrieval):

    """
    Sequential mixture of experts (MoE).

     The sequential layers are not intended to be generative / trained
     causally. Windows of historical items of a fixed size should be used
      as the query model inputs, and a single item as the label (candidate).

    Inspired by: https://arxiv.org/pdf/1902.08588.pdf
    """

    def __init__(self, query_id, candidate_id, embedding_dim=32,
                 query_context_features=None, candidate_context_features=None,
                 name="SelfAttentionMixtureOfExpertsRetrieval", **kwargs):

        self._embedding_dim = embedding_dim
        self._hash_embedding_dim = 128

        # Query model is a sequential mixture -- see model for details
        query_model = SequentialDeepHashEmbeddingMixtureOfExperts(
                hash_embedding_dim=self._hash_embedding_dim,
                embedding_dim=self._embedding_dim,
                masking=True
        )

        # Candidate model is a simple (non-sequential) hash embedder + FF
        candidate_model = DeepHashEmbedding(
                hash_embedding_dim=self._hash_embedding_dim,
                embedding_dim=self._embedding_dim
        )

        super().__init__(
                query_model=query_model,
                candidate_model=candidate_model,
                query_id=query_id,
                candidate_id=candidate_id,
                query_context_features=query_context_features,
                candidate_context_features=candidate_context_features,
                name=name,
                **kwargs
        )
