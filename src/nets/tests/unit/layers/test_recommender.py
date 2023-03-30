
import unittest
import numpy as np

from nets.layers.recommender import StringEmbedding, HashEmbedding
from nets.tests.utils import *


class TestStringEmbedding(unittest.TestCase):

    def setUp(self):
        """
        Create fresh default params for each test.
        """
        self._vocab = np.array(["one", "two", "three", "four", "five"])
        self._embedding_dim = 2

        self._default_config = {
            "vocab": self._vocab,
            "embedding_dim": self._embedding_dim
        }

    @try_except_assertion_decorator
    def test_build_with_constructor_defaults(self):
        _ = StringEmbedding(vocab=self._vocab)

    @try_except_assertion_decorator
    def test_build(self):
        _ = StringEmbedding(
                vocab=self._vocab, embedding_dim=self._embedding_dim
        )

    @try_except_assertion_decorator
    def test_build_from_config(self):
        _ = StringEmbedding.from_config(self._default_config)

    def test_build_get_config(self):
        embedder = StringEmbedding(**self._default_config)
        c = embedder.get_config()
        assert all([k in c for k, v in self._default_config.items()]), \
            """
            Missing a passed model param in the model config definition.
            Passed configuration: {}
            Returned configuration: {}
            """.format(self._default_config, c)

    def test_call(self):
        embedder = StringEmbedding(
                vocab=self._vocab, embedding_dim=self._embedding_dim
        )
        embedding = embedder(np.array(["one"]))
        assert np.array(embedding).shape[-1] == self._embedding_dim, \
            "Embedded dimension does not match configured value."


class TestHashEmbedding(unittest.TestCase):

    def setUp(self):
        """
        Create fresh default params for each test.
        """
        self._vocab = np.array(["one", "two", "three", "four", "five"])

        self._hash_bins = 10
        self._embedding_dim = 2

        self._default_config = {
            "hash_bins": self._hash_bins,
            "embedding_dim": self._embedding_dim
        }

    @try_except_assertion_decorator
    def test_build_with_constructor_defaults(self):
        _ = HashEmbedding()

    @try_except_assertion_decorator
    def test_build(self):
        _ = HashEmbedding(
                hash_bins=self._hash_bins, embedding_dim=self._embedding_dim
        )

    @try_except_assertion_decorator
    def test_build_from_config(self):
        _ = HashEmbedding.from_config(self._default_config)

    def test_build_get_config(self):
        embedder = HashEmbedding(**self._default_config)
        c = embedder.get_config()
        assert all([k in c for k, v in self._default_config.items()]), \
            """
            Missing a passed model param in the model config definition.
            Passed configuration: {}
            Returned configuration: {}
            """.format(self._default_config, c)

    def test_call(self):
        embedder = HashEmbedding.from_config(self._default_config)
        embedding = embedder(np.array(["one"]))
        assert np.array(embedding).shape[-1] == self._embedding_dim, \
            "Embedded dimension does not match configured value."