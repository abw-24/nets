
import unittest
import numpy as np

from nets.models.recommender.embedding import DeepHashEmbedding

from nets.tests.utils import try_except_assertion_decorator


class TestDeepHashEmbedding(unittest.TestCase):

    def setUp(self):
        """
        Create fresh default params for each test.
        """
        self._vocab = np.array(["one", "two", "three", "four", "five", "six"])

        self._hash_bins = 10
        self._hash_embedding_dim = 4
        self._embedding_dim = 2

        self._default_config = {
            "hash_bins": self._hash_bins,
            "hash_embedding_dim": self._hash_embedding_dim,
            "embedding_dim": self._embedding_dim,
            "hidden_dims": [3],
            "activation": "relu"
        }

    @try_except_assertion_decorator
    def test_build_with_constructor_defaults(self):
        _ = DeepHashEmbedding()

    @try_except_assertion_decorator
    def test_build(self):
        _ = DeepHashEmbedding(
                hash_bins=self._hash_bins, embedding_dim=self._embedding_dim
        )

    @try_except_assertion_decorator
    def test_build_with_constructor_kwargs(self):
        hidden_dims = [4]
        activation = "sigmoid"
        model = DeepHashEmbedding(
                hash_bins=self._hash_bins, embedding_dim=self._embedding_dim,
                hidden_dims=hidden_dims, activation=activation
        )

        assert model._dense_config["hidden_dims"] == hidden_dims
        assert model._dense_config["activation"] == activation

    @try_except_assertion_decorator
    def test_build_from_config(self):
        _ = DeepHashEmbedding.from_config(self._default_config)

    def test_get_config(self):
        embedder = DeepHashEmbedding(**self._default_config)
        c = embedder.get_config()
        assert all([k in c for k, v in self._default_config.items()]), \
            """
            Missing a passed model param in the model config definition.
            Passed configuration: {}
            Returned configuration: {}
            """.format(self._default_config, c)

    def test_call(self):
        embedder = DeepHashEmbedding.from_config(self._default_config)
        embedding = embedder(np.array(["one"]))
        assert np.array(embedding).shape[-1] == self._embedding_dim, \
            "Embedded dimension does not match configured value."
