
import unittest
import numpy as np
import tensorflow as tf

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
            "embedding_dim": self._embedding_dim,
            "position_encoding": None
        }

    @try_except_assertion_decorator
    def test_build_with_constructor_defaults(self):
        _ = StringEmbedding(vocab=self._vocab)

    @try_except_assertion_decorator
    def test_build(self):
        _ = StringEmbedding(**self._default_config)

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
        assert embedding.shape.as_list()[-1] == self._embedding_dim, \
            "Embedded dimension does not match configured value."

    def test_call_with_positional_encoding(self):
        self._default_config.update({
            "positional_encoding": "bert",
            "max_length": 100
        })
        embedder = StringEmbedding(**self._default_config)
        embedding = embedder((np.array(["one"]), None))
        assert embedding.shape.as_list()[-1] == self._embedding_dim, \
            "Embedded dimension does not match configured value."

    def test_call_with_relative_positional_encoding(self):
        self._default_config.update({
            "positional_encoding": "relative"
        })
        embedder = StringEmbedding(**self._default_config)
        embedding = embedder((np.array(["one"]), None))
        assert embedding.shape.as_list()[-1] == self._embedding_dim, \
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
        assert embedding.shape.as_list()[-1] == self._embedding_dim, \
            "Embedded dimension does not match configured value."

    def test_call_with_positional_encoding(self):
        self._default_config.update({
            "positional_encoding": "bert",
            "max_length": 100
        })
        embedder = HashEmbedding(**self._default_config)
        embedding = embedder((np.array(["one"]), None))
        assert embedding.shape.as_list()[-1] == self._embedding_dim, \
            "Embedded dimension does not match configured value."

    def test_call_with_relative_positional_encoding(self):
        self._default_config.update({"positional_encoding": "relative"})
        embedder = HashEmbedding(**self._default_config)
        embedding = embedder((np.array(["one"]), None))
        assert embedding.shape.as_list()[-1] == self._embedding_dim, \
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
        assert embedding.shape.as_list()[-1] == self._embedding_dim, \
            "Embedded dimension does not match configured value."


class TestSequentialDeepHashEmbeddingWithGRU(unittest.TestCase):

    def setUp(self):
        """
        Create fresh default params for each test.
        """
        self._default_call_inputs = (
            np.array([[1,2,3], [4,5,6]]),
            None
        )

        self._hash_bins = 10
        self._hash_embedding_dim = 4
        self._embedding_dim = 2
        self._masking = False

        self._default_config = {
            "hash_bins": self._hash_bins,
            "hash_embedding_dim": self._hash_embedding_dim,
            "embedding_dim": self._embedding_dim,
            "masking": self._masking
        }

    @try_except_assertion_decorator
    def test_build_with_constructor_defaults(self):
        _ = SequentialDeepHashEmbeddingWithGRU()

    @try_except_assertion_decorator
    def test_build(self):
        _ = SequentialDeepHashEmbeddingWithGRU(
                hash_bins=self._hash_bins, embedding_dim=self._embedding_dim
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
        assert tuple(embedding.shape.as_list()) == \
               (self._default_call_inputs[0].shape[0], self._embedding_dim), \
            "Embedded output does not match expected shape."

    def test_call_with_masked_input(self):
        self._default_config.update({"masking": True})
        unpadded = [[1,2,3], [4,5,6,7]]
        padded = tf.keras.preprocessing.sequence.pad_sequences(
                unpadded, padding="post"
        )
        embedder = SequentialDeepHashEmbeddingWithGRU(**self._default_config)
        embedding = embedder((padded, None))
        assert tuple(embedding.shape.as_list()) == \
               (padded.shape[0], self._embedding_dim), \
            "Embedded output does not match expected shape."


class TestSequentialDeepHashEmbeddingWithAttention(unittest.TestCase):

    def setUp(self):
        """
        Create fresh default params for each test.
        """
        self._default_call_inputs = (
            np.array([["one", "two", "three"],
                      ["four", "five", "six"]]),
            None
        )

        self._hash_bins = 10
        self._hash_embedding_dim = 4
        self._embedding_dim = 2
        self._attention_heads = 4
        self._attention_key_dim = 128
        self._attention_concat = False
        self._attention_mask = False
        self._masking = False

        self._default_config = {
            "hash_bins": self._hash_bins,
            "hash_embedding_dim": self._hash_embedding_dim,
            "embedding_dim": self._embedding_dim,
            "attention_heads": self._attention_heads,
            "attention_key_dim": self._attention_key_dim,
            "attention_concat": self._attention_concat,
            "attention_causal_mask": self._attention_mask,
            "masking": self._masking
        }

    @try_except_assertion_decorator
    def test_build_with_constructor_defaults(self):
        _ = SequentialDeepHashEmbeddingWithAttention()

    @try_except_assertion_decorator
    def test_build(self):
        _ = SequentialDeepHashEmbeddingWithAttention(
                hash_bins=self._hash_bins,
                embedding_dim=self._embedding_dim,
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
        # Since the default is to not concatenate, the output shape and
        # dimensions should be the same
        assert len(embedding.shape.as_list()) == 3, "Unexpected shape."
        assert tuple(embedding.shape.as_list()[:-1]) == self._default_call_inputs[0].shape, \
            "Shapes do not match.\nOutput: {}\nInput:{}"\
                .format(tuple(embedding.shape.as_list()[:-1]),
                        self._default_call_inputs[0].shape)

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
        assert len(embedding.shape.as_list()) == 2, "Unexpected shape."
        assert embedding.shape.as_list()[-1] == self._embedding_dim, \
            "Flattened model dim not matching expectations: {}"\
                .format(embedding.shape.as_list()[-1])

    def test_call_with_pooling(self):
        self._default_config.update({
            "attention_concat": False, "attention_pooling": True
        })
        embedder = SequentialDeepHashEmbeddingWithAttention(
                **self._default_config
        )
        embedding = embedder(self._default_call_inputs)
        # If pooling is True, then we should have averaged over the timestep
        # dimension, and again have 2 dims with final dim == embedding_dim
        assert len(embedding.shape.as_list()) == 2, "Unexpected shape."
        assert embedding.shape.as_list()[-1] == self._embedding_dim, \
            "Flattened model dim not matching expectations: {}"\
                .format(embedding.shape.as_list()[-1])

    def test_call_with_masked_input(self):
        self._default_config.update({"masking": True})
        unpadded = [[1,2,3], [4,5,6,7]]
        padded = tf.keras.preprocessing.sequence.pad_sequences(
                unpadded, padding="post"
        )
        embedder = SequentialDeepHashEmbeddingWithAttention(**self._default_config)
        embedding = embedder((padded, None))

        # With default call behavior, the steps dim should now be 4 with the
        # padded inputs, and otherwise be the same
        assert len(embedding.shape.as_list()) == 3, "Unexpected shape."
        assert embedding.shape.as_list()[1] == 4, "Output steps not padded?"
        assert embedding.shape.as_list()[-1] == self._embedding_dim, \
            "Embedded dimension does not match configured value."
        # Check to see that the mask propagated
        assert embedding._keras_mask is not None, "Mask did not propagate."
        assert embedder.compute_mask((padded, None))[0, -1] == False, \
            "Mask not propagated/expanded as expected: {}"\
                .format(embedding._embedding._keras_mask.numpy())


class TestSequentialDeepHashEmbeddingMixtureOfExperts(unittest.TestCase):

    def setUp(self):
        """
        Create fresh default params for each test.
        """
        self._default_call_inputs = (
            np.array([["one", "two", "three"],
                      ["four", "five", "six"]]),
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
        assert len(embedding.shape.as_list()) == 2, "Unexpected shape."
        assert embedding.shape.as_list()[-1] == self._embedding_dim, \
            "Embedded dimension does not match configured value."
