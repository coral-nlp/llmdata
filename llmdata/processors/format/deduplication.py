from itertools import tee
from typing import Any, Iterable

import mmh3
import numpy as np
import ray
from pydantic import Field
from ray import remote
from ray.types import ObjectRef

from llmdata.core.ops import MapFn, Row
from llmdata.core.registry import components
from llmdata.core.utils import get_field, set_field


class Signature:
    """BloomLSH signature."""

    def __init__(
        self,
        threshold: float,
        num_permutations: int,
        shingle_word_count: int,
        num_bands: int,
        band_size: int,
        seed: int = 1854201893,
        prime: int = 4294967311,
    ) -> None:
        self.threshold = threshold
        self.num_permutations = num_permutations
        self.ngrams = shingle_word_count
        self.num_bands = num_bands
        self.band_size = band_size
        self.num_hashes = self.num_bands * self.band_size
        self.max_val = (1 << 32) - 1  # Largest possible hash value, uint32
        self.prime = prime
        # Set up permutation vectors
        np.random.seed(seed)
        self.a = np.random.randint(1, self.max_val, self.num_permutations, dtype=np.uint64)
        self.b = np.random.randint(0, self.max_val, self.num_permutations, dtype=np.uint64)

    def _get_ngrams(self, sequence: list[str]) -> Iterable:
        """Construct ngrams for a given list of tokens. Directly taken from nltk package to avoid dependency."""
        iterables = tee(sequence, self.ngrams)
        for i, sub_iterable in enumerate(iterables):
            for _ in range(i):
                next(sub_iterable, None)
        return zip(*iterables, strict=False)

    def _get_shingles(self, text: str) -> set[str]:
        """Split text into ngram shingles."""
        words = text.split()
        if len(words) < self.ngrams:
            return {
                " ".join(words),
            }
        else:
            return {" ".join(ng) for ng in self._get_ngrams(words)}

    def _get_minhash_signature(self, shingles: set[str]) -> "np.ndarray":
        """Compute the MinHash signature for a given text."""
        # Return 0 if empty shingle set
        if not shingles:
            return np.zeros(self.num_permutations, dtype=np.uint64)
        # Initialize signature with maximum values
        signature = np.full(self.num_permutations, self.max_val, dtype=np.uint64)
        # Update signature with each shingle
        for shingle in shingles:
            shingle_hash = mmh3.hash(shingle.encode("utf-8"), signed=False)
            hashes = ((self.a * shingle_hash + self.b) % self.prime) % self.max_val
            signature = np.minimum(signature, hashes)
        return signature

    def _get_band_signature(self, minhash_signature: "np.ndarray") -> "np.ndarray":
        """Compute the band signature for a given minhash signature."""
        # Split into bands
        bands = np.array_split(minhash_signature, self.num_bands)
        # Compute each bands' individual hashes
        band_hashes = np.zeros(shape=(self.num_bands,), dtype=np.uint64)
        for band_idx, band in enumerate(bands):
            for hash in band:
                band_hashes[band_idx] += mmh3.hash(hash.tobytes(), signed=False)
            band_hashes[band_idx] = band_hashes[band_idx] % self.max_val

        return band_hashes

    def __call__(self, text: str) -> "np.ndarray":
        """Compute the BloomLSH signature for a given text."""
        # Compute shingles
        shingles = self._get_shingles(text)
        # Compute minhash signature of shingle set
        signature = self._get_minhash_signature(shingles)
        # Compute Bloom bands
        bands = self._get_band_signature(signature)
        return bands.astype(np.uint32)  # Ensure uint32 for bloom filter


class BandedBloomFilter:
    """Implementation of a bloom filter specifically for BloomLSH.

    TODO: This currently does not scale for large-ish datasets.
    """

    def __init__(
        self,
        bloom_size: int,
        bloom_hashes: int,
        lsh_permutations: int,
        lsh_threshold: float,
        lsh_ngram_size: int,
        lsh_seed: int = 1854201893,
        lsh_prime: int = 4294967311,
    ) -> None:
        # Parameters for MinHash Signature
        self.lsh_permutations = lsh_permutations
        self.lsh_threshold = lsh_threshold
        self.lsh_ngram_size = lsh_ngram_size
        # Parameters for bloom filter
        self.bloom_size = bloom_size
        self.bloom_hashes = bloom_hashes
        # Shared parameters
        self.num_bands, self.band_size = self._set_bands()
        # Signature provider
        self.signature = Signature(
            threshold=lsh_threshold,
            num_permutations=lsh_permutations,
            shingle_word_count=lsh_ngram_size,
            num_bands=self.num_bands,
            band_size=self.band_size,
            seed=lsh_seed,
            prime=lsh_prime,
        )
        # Filter state
        self.state = np.zeros((self.num_bands, self.bloom_size), dtype=bool)

    def _set_bands(self) -> tuple[int, int]:
        """Calculate optimal band number and band size for given similarity threshold."""
        best_b, best_r = 1, self.lsh_permutations
        best_error = float("inf")

        for b in range(1, self.lsh_permutations + 1):
            if self.lsh_permutations % b == 0:
                r = self.lsh_permutations // b
                estimated_threshold = (1.0 / b) ** (1.0 / r)
                error = abs(estimated_threshold - self.lsh_threshold)

                if error < best_error:
                    best_error = error
                    best_b, best_r = b, r

        return best_b, best_r

    def _hashes(self, item: np.uint32) -> list[int]:
        return [mmh3.hash(item.tobytes(), i) % self.bloom_size for i in range(self.bloom_hashes)]

    def _signature(self, val: str) -> "np.ndarray":
        """Compute the signature for an incoming string."""
        return self.signature(val)

    def put(self, data: str) -> None:
        """Add a signature to the bloom filter."""
        band_values = self._signature(data)
        for band_idx, value in enumerate(band_values):
            hash_indices = self._hashes(value)
            self.state[band_idx, hash_indices] = True

    def get(self, data: str) -> bool:
        """Check bloom filter for matches on given signature."""
        # Return first matching signature or None
        band_values = self._signature(data)
        for band_idx, value in enumerate(band_values):
            hash_indices = self._hashes(value)
            if np.all(self.state[band_idx, hash_indices]):
                return True  # type: ignore
        return False


@remote
class BandedBloomFilterActor(BandedBloomFilter):
    """Distributed wrapper for banded bloom filter."""

    def __repr__(self) -> str:
        """Summarizes bloom filter state for ray logging."""
        return f"BandedBloomFilter(Bits: {self.num_bands}x{self.bloom_size}, Hashes: {self.bloom_hashes}, Threshold: {self.lsh_threshold})"


def _get_or_create_actor(name: str, memory: int, **kwargs: Any) -> "ObjectRef[BandedBloomFilterActor]":
    return BandedBloomFilterActor.options(name=name, memory=memory, num_cpus=1.0, get_if_exists=True).remote(**kwargs)  # type: ignore[attr-defined]


@components.add("format", "deduplication")
class DeduplicationFormatter(MapFn):
    """Formatter to deduplicate the text of rows across the whole dataset on paragraph level.

    Uses a BloomLSH filter internally to detect similar content: https://arxiv.org/pdf/2411.04257v1
    """

    name: str = Field(default="deduplication_formatter", description="Name of the formatter")
    on: str = Field(default="text", description="Column to read text fro")
    to: str = Field(default="text", description="Column to save deduplicated result text to")
    bloom_size: int = Field(default=1_000_000, description="Size of the bloom filter", gt=0)
    bloom_hashes: int = Field(default=3, description="Number of hash functions for bloom filter", gt=0)
    lsh_threshold: float = Field(
        default=0.8, description="Jaccard similarity threshold for deduplication", ge=0.0, le=1.0
    )
    lsh_permutations: int = Field(default=256, description="Number of permutations for MinHash", gt=0)
    lsh_ngram_size: int = Field(default=8, description="Word count for ngram shingles", gt=0)
    split_char: str = Field(default="\n", description="Character to split text into paragraphs")
    memory: int = Field(default=256, description="Memory to reserve for the bloom filter in MB (default 256MB)", gt=0)

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self._actor = _get_or_create_actor(
            name="bloom_actor",
            memory=self.memory * 1024 * 1024,  # MB to byte
            bloom_size=self.bloom_size,
            bloom_hashes=self.bloom_hashes,
            lsh_permutations=self.lsh_permutations,
            lsh_threshold=self.lsh_threshold,
            lsh_ngram_size=self.lsh_ngram_size,
        )

    def __call__(self, row: Row) -> Row:
        """Single formatting step.

        Decompose the document into paragraphs. For each paragraph, compute a signature and check if it matches the bloom
        filter. If not, insert it and continue. Otherwise, mark as duplicate and continue. Finally, re-assemble the document
        text only with the non-duplicate parts.

        As the underlying bloom filter updates sequentially, no duplicated text will remain after execution.
        """
        text = get_field(row, self.on)
        if not text:
            return row
        paragraphs = text.split(self.split_char) if self.split_char is not None else [text]
        unique = []
        for paragraph in paragraphs:
            match = ray.get(self._actor.get.remote(paragraph))
            if match is None:
                # No matches - insert it and keep it
                ray.get(self._actor.put.remote(paragraph))
                unique.append(True)
            else:
                # Matches existing - don't insert, mark as duplicate
                unique.append(False)
        if not any(unique):
            set_field(row, self.to, "")
        else:
            paragraphs = [text for (text, label) in zip(paragraphs, unique, strict=False) if label]
            if self.split_char is not None:
                set_field(row, self.to, self.split_char.join(paragraphs))
            else:
                set_field(row, self.to, paragraphs[0])
        return row
