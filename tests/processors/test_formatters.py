import unittest

import numpy as np

from llmdata.processors.format.deduplication import DeduplicationFormatter, Signature


class TestDeduplicationSignature(unittest.TestCase):
    """Test MinHash signature computation."""

    def setUp(self):
        """Set up test configuration and signature computer."""
        self.config_kwargs = {
            "lsh_threshold": 0.8,
            "lsh_permutations": 64,
            "ngrams": 3,
            "num_bands": 12,
            "band_size": 4,
        }
        self.signature = Signature(**self.config_kwargs, seed=420)  # Fixed lsh_seed for reproducibility

    def test_signature_initialization(self):
        """Test signature computer initialization."""
        self.assertEqual(self.signature.lsh_permutations, self.config_kwargs["lsh_permutations"])
        self.assertEqual(self.signature.ngrams, self.config_kwargs["ngrams"])
        self.assertEqual(self.signature.num_bands, self.config_kwargs["num_bands"])
        self.assertEqual(self.signature.band_size, self.config_kwargs["band_size"])

        # Check permutation vectors are initialized
        self.assertEqual(len(self.signature.a), self.config_kwargs["lsh_permutations"])
        self.assertEqual(len(self.signature.b), self.config_kwargs["lsh_permutations"])
        self.assertTrue(np.all(self.signature.a > 0))  # All values should be positive

    def test_shingle_generation(self):
        """Test character-level shingle generation."""
        # Normal case
        shingles = self.signature._get_shingles("hello")
        expected = {"hel", "ell", "llo"}
        self.assertEqual(shingles, expected)

        # Short text
        shingles = self.signature._get_shingles("hi")
        expected = {"hi"}
        self.assertEqual(shingles, expected)

        # Empty text
        shingles = self.signature._get_shingles("")
        expected = {""}
        self.assertEqual(shingles, expected)

        # Exact length
        shingles = self.signature._get_shingles("abc")  # ngrams = 3
        expected = {"abc"}
        self.assertEqual(shingles, expected)

    def test_minhash_signature_properties(self):
        """Test MinHash signature computation properties."""
        text = "The quick brown fox jumps over the lazy dog"
        shingles = self.signature._get_shingles(text)

        minhash_sig = self.signature._get_minhash_signature(shingles)

        # Check output shape and type
        self.assertEqual(minhash_sig.shape, (64,))
        self.assertEqual(minhash_sig.dtype, np.uint64)

        # All values should be less than max_val
        self.assertTrue(np.all(minhash_sig < self.signature.max_val))

        # Should not be all zeros (unless empty input)
        self.assertFalse(np.all(minhash_sig == 0))

    def test_minhash_empty_input(self):
        """Test MinHash behavior with empty input."""
        empty_shingles = set()
        minhash_sig = self.signature._get_minhash_signature(empty_shingles)

        # Should return all zeros
        expected = np.zeros(64, dtype=np.uint64)
        np.testing.assert_array_equal(minhash_sig, expected)

    def test_band_signature_properties(self):
        """Test band signature computation."""
        text = "The quick brown fox jumps over the lazy dog"
        shingles = self.signature._get_shingles(text)
        minhash_sig = self.signature._get_minhash_signature(shingles)

        band_sig = self.signature._get_band_signature(minhash_sig)

        # Check output shape and type
        self.assertEqual(band_sig.shape, (self.config_kwargs["num_bands"],))
        self.assertEqual(band_sig.dtype, np.uint64)

        # All values should be less than max_val
        self.assertTrue(np.all(band_sig < self.signature.max_val))

    def test_signature_call_interface(self):
        """Test the main signature computation interface."""
        text = "The quick brown fox jumps over the lazy dog"
        signature = self.signature(text)

        # Check output properties
        self.assertEqual(signature.shape, (self.config_kwargs["num_bands"],))
        self.assertEqual(signature.dtype, np.uint32)

        # Should be consistent across calls
        signature2 = self.signature(text)
        np.testing.assert_array_equal(signature, signature2)

    def test_signature_similarity(self):
        """Test that similar texts produce similar signatures."""
        text1 = "The quick brown fox jumps over the lazy dog"
        text2 = "The quick brown fox jumps over the lazy cat"  # Similar
        text3 = "Completely different text with no overlap"  # Different

        sig1 = self.signature(text1)
        sig2 = self.signature(text2)
        sig3 = self.signature(text3)

        # Similar texts should have some matching bands
        matches_12 = np.sum(sig1 == sig2)
        matches_13 = np.sum(sig1 == sig3)

        # Similar texts should have more matches than different texts
        self.assertEqual(matches_13, 0)
        self.assertGreaterEqual(matches_12, matches_13)

    def test_signature_deterministic(self):
        """Test that signatures are deterministic with same lsh_seed."""
        sig1 = Signature(**self.config_kwargs, seed=42)
        sig2 = Signature(**self.config_kwargs, seed=42)

        text = "Test text for deterministic signatures"

        result1 = sig1(text)
        result2 = sig2(text)

        np.testing.assert_array_equal(result1, result2)

    def test_signature_different_seeds(self):
        """Test that different seeds produce different signatures."""
        sig1 = Signature(**self.config_kwargs, seed=42)
        sig2 = Signature(**self.config_kwargs, seed=12345)

        text = "Test text for different seeds"

        result1 = sig1(text)
        result2 = sig2(text)

        # Should be different (with very high probability)
        self.assertFalse(np.array_equal(result1, result2))


class TestDeduplicationFormatter(unittest.TestCase):
    """Tests for DeduplicationFormatter."""

    def setUp(self):
        """Set up test configuration and bloom filter."""
        self.config_kwargs = {
            "lsh_threshold": 0.8,
            "lsh_permutations": 64,
            "ngrams": 3,
            "bloom_size": 10_000,
            "bloom_hashes": 3,
            "use_distributed_actor": False,
        }
        self.dedup = DeduplicationFormatter(**self.config_kwargs)

    def test_band_optimization(self):
        """Test that band optimization produces reasonable results."""
        estimated_threshold = (1.0 / self.dedup.num_bands) ** (1.0 / self.dedup.band_size)
        error = abs(estimated_threshold - self.dedup.lsh_threshold)

        # Should be reasonably close to target lsh_threshold
        self.assertLess(error, 0.2, f"Band optimization error too large: {error}")

    def test_band_thresholds(self):
        """Test band optimization for different thresholds."""
        thresholds = [0.7, 0.8, 0.9, 0.95]

        for threshold in thresholds:
            config = self.config_kwargs.copy()
            config["lsh_threshold"] = threshold
            dedup = DeduplicationFormatter(**self.config_kwargs)

            # Check bands are valid
            self.assertGreater(dedup.num_bands, 0)
            self.assertGreater(dedup.band_size, 0)
            self.assertEqual(dedup.num_bands * dedup.band_size, config["lsh_permutations"])

            # Check lsh_threshold approximation
            estimated = (1.0 / dedup.num_bands) ** (1.0 / dedup.band_size)
            error = abs(estimated - threshold)
            self.assertLess(error, 0.2, f"Band optimization error too large: {error}")

    def test_call_no_documents(self):
        """Test realistic deduplication using the complete supplied pipeline."""
        dedup = DeduplicationFormatter(
            **{
                "lsh_threshold": 0.8,
                "lsh_permutations": 64,
                "ngrams": 3,
                "bloom_size": 10_000,
                "bloom_hashes": 3,
                "use_distributed_actor": False,
                "split_char": None,
            }
        )

        base_texts = [
            "Machine learning is a subset of artificial intelligence",
            "Deep learning uses neural networks with multiple layers",
            "Natural language processing helps computers understand text",
            "Computer vision enables machines to interpret visual information",
            "Reinforcement learning trains agents through rewards and penalties",
        ]
        all_texts = []
        all_texts.extend(base_texts)  # Original texts
        all_texts.extend(base_texts[:3])  # Duplicate first 3
        all_texts.extend(
            [  # Slight variations
                "Machine learning is a subset of artificial intelligence.",  # Added period
                "Deep learning uses neural networks with many layers",  # "multiple" -> "many"
            ]
        )
        # Mock rows
        all_texts = [{"id": i, "text": text} for i, text in enumerate(all_texts)]

        processed_texts = list(filter(lambda x: x["text"] != "", map(dedup, all_texts)))

        original_ids = {x["id"] for x in all_texts}
        deduplicated_ids = {x["id"] for x in processed_texts}
        # Verify deduplication worked
        self.assertLess(len(deduplicated_ids), len(original_ids))
        self.assertEqual(deduplicated_ids, {0, 1, 2, 3, 4, 9})

    def test_call_paragraphs(self):
        """Test realistic deduplication using the complete supplied pipeline."""
        dedup = DeduplicationFormatter(
            **{
                "lsh_threshold": 0.8,
                "lsh_permutations": 64,
                "ngrams": 8,
                "bloom_size": 10_000,
                "bloom_hashes": 3,
                "use_distributed_actor": False,
                "split_char": "\n",
            }
        )

        all_texts = [
            # Single sentence, no sep
            "Machine learning is a subset of artificial intelligence",
            # Partial duplicate
            "Machine learning is a subset of artificial intelligence\n Deep learning uses neural networks with multiple layers",
            # Full duplicate
            "Machine learning is a subset of artificial intelligence\n Deep learning uses neural networks with multiple layers",
            # Full duplicate, reversed order
            "Deep learning uses neural networks with multiple layers\n Machine learning is a subset of artificial intelligence",
            # Small modification, no sep
            "Machine learning is a subset of artificial intelligence.",
            # Partial duplicate, with sep
            "ML is a subset of AI\n Deep learning uses neural networks with multiple layers",
        ]
        all_texts = [{"id": i, "text": text} for i, text in enumerate(all_texts)]

        processed_texts = list(map(dedup, all_texts))
        processed_texts = list(filter(lambda x: x["text"] != "", processed_texts))

        original_ids = {x["id"] for x in all_texts}
        deduplicated_ids = {x["id"] for x in processed_texts}
        # Verify deduplication worked
        self.assertLess(len(deduplicated_ids), len(original_ids))
        self.assertEqual(deduplicated_ids, {0, 1, 5})
