
import unittest
import numpy as np

from nets.models.recommender.embedding import \
    StringEmbedding, HashEmbedding, DeepHashEmbedding, \
    SequentialDeepHashEmbeddingWithAttention, \
    SequentialDeepHashEmbeddingWithGRU, \
    SequentialDeepHashEmbeddingMixtureOfExperts

from nets.tests.utils import try_except_assertion_decorator


#TODO: Add tests to cover context features

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
        embedder = StringEmbedding(**self._default_config)
        embedding = embedder((np.array(["one"]), None))
        assert np.array(embedding).shape[-1] == self._embedding_dim, \
            "Embedded dimension does not match configured value."


class TestHashEmbedding(unittest.TestCase):

    def setUp(self):
        """
        Create fresh default params for each test.
        """

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
        embedder = HashEmbedding(**self._default_config)
        embedding = embedder((np.array(["one"]), None))
        assert np.array(embedding).shape[-1] == self._embedding_dim, \
            "Embedded dimension does not match configured value."


class TestDeepHashEmbedding(unittest.TestCase):

    def setUp(self):
        """
        Create fresh default params for each test.
        """

        self._hash_bins = 10
        self._hash_embedding_dim = 4
        self._embedding_dim = 2

        self._default_config = {
            "hash_bins": self._hash_bins,
            "hash_embedding_dim": self._hash_embedding_dim,
            "embedding_dim": self._embedding_dim
        }

    @try_except_assertion_decorator
    def test_build_with_constructor_defaults(self):
        _ = DeepHashEmbedding()

    @try_except_assertion_decorator
    def test_build(self):
        _ = DeepHashEmbedding(
                hash_bins=self._hash_bins, embedding_dim=self._embedding_dim
        )

    def test_build_with_feedforward_kwargs(self):
        ff_config = {
            "hidden_dims": [4],
            "activation": "sigmoid"
        }
        model = DeepHashEmbedding(
                hash_bins=self._hash_bins, embedding_dim=self._embedding_dim,
                feedforward_config=ff_config
        )
        assert all([model.feedforward_config[k] == v for k, v in ff_config.items()])

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
        embedder = DeepHashEmbedding(**self._default_config)
        embedding = embedder((np.array(["one"]), None))
        assert np.array(embedding).shape[-1] == self._embedding_dim, \
            "Embedded dimension does not match configured value."


class TestSequentialDeepHashEmbeddingWithGRU(unittest.TestCase):

    def setUp(self):
        """
        Create fresh default params for each test.
        """
        self._default_call_inputs = (
            np.array([["one", "two", "three"], ["four", "five", "six"]]),
            None
        )

        self._hash_bins = 10
        self._hash_embedding_dim = 4
        self._embedding_dim = 2
        self._last_n = None

        self._default_config = {
            "hash_bins": self._hash_bins,
            "hash_embedding_dim": self._hash_embedding_dim,
            "embedding_dim": self._embedding_dim,
            "last_n": None
        }

    @try_except_assertion_decorator
    def test_build_with_constructor_defaults(self):
        _ = SequentialDeepHashEmbeddingWithGRU()

    @try_except_assertion_decorator
    def test_build(self):
        _ = SequentialDeepHashEmbeddingWithGRU(
                hash_bins=self._hash_bins, embedding_dim=self._embedding_dim,
                last_n=self._last_n
        )

    @try_except_assertion_decorator
    def test_build_from_config(self):
        _ = SequentialDeepHashEmbeddingWithGRU.from_config(self._default_config)

    def test_get_config(self):
        embedder = SequentialDeepHashEmbeddingWithGRU(**self._default_config)
        c = embedder.get_config()
        assert all([k in c for k, v in self._default_config.items()]), \
            """
            Missing a passed model param in the model config definition.
            Passed configuration: {}
            Returned configuration: {}
            """.format(self._default_config, c)

    def test_call(self):
        embedder = SequentialDeepHashEmbeddingWithGRU(**self._default_config)
        embedding = embedder(self._default_call_inputs)
        assert np.array(embedding).shape[-1] == self._embedding_dim, \
            "Embedded dimension does not match configured value."

    def test_call_with_last_n(self):
        # Update default config with last_n value
        self._default_config.update({"last_n": 1})
        embedder = SequentialDeepHashEmbeddingWithGRU(**self._default_config)
        embedding = embedder(self._default_call_inputs)
        assert np.array(embedding).shape[-1] == self._embedding_dim, \
            "Embedded dimension does not match configured value."


class TestSequentialDeepHashEmbeddingWithAttention(unittest.TestCase):

    def setUp(self):
        """
        Create fresh default params for each test.
        """
        self._default_call_inputs = (
            np.array([["one", "two", "three"], ["four", "five", "six"]]),
            None
        )

        self._hash_bins = 10
        self._hash_embedding_dim = 4
        self._embedding_dim = 2
        self._last_n = None
        self._attention_heads = 4
        self._attention_key_dim = 128
        self._attention_concat = False
        self._attention_mask = False

        self._default_config = {
            "hash_bins": self._hash_bins,
            "hash_embedding_dim": self._hash_embedding_dim,
            "embedding_dim": self._embedding_dim,
            "last_n": None,
            "attention_heads": self._attention_heads,
            "attention_key_dim": self._attention_key_dim,
            "attention_concat": self._attention_concat,
            "attention_mask": self._attention_mask
        }

    @try_except_assertion_decorator
    def test_build_with_constructor_defaults(self):
        _ = SequentialDeepHashEmbeddingWithAttention()

    @try_except_assertion_decorator
    def test_build(self):
        _ = SequentialDeepHashEmbeddingWithAttention(
                hash_bins=self._hash_bins,
                embedding_dim=self._embedding_dim,
                last_n=self._last_n,
                attention_key_dim=self._attention_key_dim,
                attention_mask=self._attention_mask,
                attention_heads=self._attention_heads,
                attention_concat=self._attention_concat
        )

    @try_except_assertion_decorator
    def test_build_from_config(self):
        _ = SequentialDeepHashEmbeddingWithAttention.from_config(
                self._default_config
        )

    def test_get_config(self):
        embedder = SequentialDeepHashEmbeddingWithAttention(
                **self._default_config
        )
        c = embedder.get_config()
        assert all([k in c for k, v in self._default_config.items()]), \
            """
            Missing a passed model param in the model config definition.
            Passed configuration: {}
            Returned configuration: {}
            """.format(self._default_config, c)

    def test_call(self):
        embedder = SequentialDeepHashEmbeddingWithAttention(
                **self._default_config
        )
        embedding = embedder(self._default_call_inputs)
        # Since the default is to not concatenate, the second dimension
        # should have the same dimension as the sequence length
        assert len(np.array(embedding).shape) == 3, "Unexpected shape."
        assert np.array(embedding).shape[-1] == self._embedding_dim, \
            "Embedded dimension does not match configured value."
        assert np.array(embedding).shape[1] == self._default_call_inputs[0].shape[1], \
            "Timestep dimension does not match expected value of {}."\
                .format(self._default_call_inputs[0].shape[1])

    def test_call_with_last_n(self):
        # Update default config with last_n value
        last_n = 2
        self._default_config.update({"last_n": last_n})
        embedder = SequentialDeepHashEmbeddingWithAttention(
                **self._default_config
        )
        embedding = embedder(self._default_call_inputs)
        # If last_n is used with attention_concat = False, we should get
        # just last_n many timesteps in the output
        assert len(np.array(embedding).shape) == 3, "Unexpected shape."
        assert np.array(embedding).shape[-1] == self._embedding_dim, \
            "Embedded dimension does not match configured value."
        assert np.array(embedding).shape[1] == last_n, \
            "Timestep dimension does not match expected value of {}."\
                .format(last_n)

    def test_call_with_concat(self):
        self._default_config.update({
            "attention_concat": True, "attention_pooling": False
        })
        embedder = SequentialDeepHashEmbeddingWithAttention(
                **self._default_config
        )
        embedding = embedder(self._default_call_inputs)
        # If concat is True, then we should have flattened down to 2 dims
        # and the final dim should be == embedding_dim (FF maps down)
        assert len(np.array(embedding).shape) == 2, "Unexpected shape."
        assert np.array(embedding).shape[-1] == self._embedding_dim, \
            "Flattened model dim not matching expectations: {}"\
                .format(np.array(embedding).shape[-1])


class TestSequentialDeepHashEmbeddingMixtureOfExperts(unittest.TestCase):

    def setUp(self):
        """
        Create fresh default params for each test.
        """
        self._default_call_inputs = (
            np.array([["one", "two", "three"], ["four", "five", "six"]]),
            None
        )

        self._hash_embedding_dim = 4
        self._embedding_dim = 2

        self._default_config = {
            "hash_embedding_dim": self._hash_embedding_dim,
            "embedding_dim": self._embedding_dim
        }

    @try_except_assertion_decorator
    def test_build_with_constructor_defaults(self):
        _ = SequentialDeepHashEmbeddingWithAttention()

    @try_except_assertion_decorator
    def test_build(self):
        _ = SequentialDeepHashEmbeddingMixtureOfExperts(
                hash_embedding_dim=self._hash_embedding_dim,
                embedding_dim=self._embedding_dim
        )

    @try_except_assertion_decorator
    def test_build_from_config(self):
        _ = SequentialDeepHashEmbeddingMixtureOfExperts.from_config(
                self._default_config
        )

    def test_get_config(self):
        embedder = SequentialDeepHashEmbeddingMixtureOfExperts(
                **self._default_config
        )
        c = embedder.get_config()
        assert all([k in c for k, v in self._default_config.items()]), \
            """
            Missing a passed model param in the model config definition.
            Passed configuration: {}
            Returned configuration: {}
            """.format(self._default_config, c)

    def test_call(self):
        embedder = SequentialDeepHashEmbeddingMixtureOfExperts(
                **self._default_config
        )
        embedding = embedder(self._default_call_inputs)
        assert len(np.array(embedding).shape) == 2, "Unexpected shape."
        assert np.array(embedding).shape[-1] == self._embedding_dim, \
            "Embedded dimension does not match configured value."
