from itertools import tee
from typing import Any, Iterable, Optional

import mmh3
import numpy as np
import ray
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
from ray import remote
from ray.types import ObjectRef

from llmdata.core.ops import MapFn, Row
from llmdata.core.registry import components
from llmdata.core.utils import get_field, set_field


class Signature:
    """BloomLSH signature."""

    def __init__(
        self,
        lsh_threshold: float,
        lsh_permutations: int,
        ngrams: int,
        num_bands: int,
        band_size: int,
        seed: int = 1854201893,
        prime: int = 4294967311,
    ) -> None:
        self.lsh_threshold = lsh_threshold
        self.lsh_permutations = lsh_permutations
        self.ngrams = ngrams
        self.num_bands = num_bands
        self.band_size = band_size
        self.num_hashes = self.num_bands * self.band_size
        self.max_val = (1 << 32) - 1
        self.prime = prime
        np.random.seed(seed)
        self.a = np.random.randint(1, self.max_val, self.lsh_permutations, dtype=np.uint64)
        self.b = np.random.randint(0, self.max_val, self.lsh_permutations, dtype=np.uint64)

    def _get_ngrams(self, sequence: list[str]) -> Iterable:
        """Construct ngrams for a given list of tokens."""
        iterables = tee(sequence, self.ngrams)
        for i, sub_iterable in enumerate(iterables):
            for _ in range(i):
                next(sub_iterable, None)
        return zip(*iterables, strict=False)

    def _get_shingles(self, text: str) -> set[str]:
        """Split text into character-level ngram shingles."""
        if len(text) < self.ngrams:
            return {text}
        return {text[i : i + self.ngrams] for i in range(len(text) - self.ngrams + 1)}

    def _get_minhash_signature(self, shingles: set[str]) -> "np.ndarray":
        """Compute the MinHash signature for a given text."""
        if not shingles:
            return np.zeros(self.lsh_permutations, dtype=np.uint64)
        signature = np.full(self.lsh_permutations, self.max_val, dtype=np.uint64)
        for shingle in shingles:
            shingle_hash = mmh3.hash(shingle.encode("utf-8"), signed=False)
            hashes = ((self.a * shingle_hash + self.b) % self.prime) % self.max_val
            signature = np.minimum(signature, hashes)
        return signature

    def _get_band_signature(self, minhash_signature: "np.ndarray") -> "np.ndarray":
        """Compute the band signature for a given minhash signature."""
        bands = np.array_split(minhash_signature, self.num_bands)
        band_hashes = np.zeros(shape=(self.num_bands,), dtype=np.uint64)
        for band_idx, band in enumerate(bands):
            for h in band:
                band_hashes[band_idx] += mmh3.hash(h.tobytes(), signed=False)
            band_hashes[band_idx] = band_hashes[band_idx] % self.max_val
        return band_hashes

    def __call__(self, text: str) -> "np.ndarray":
        """Compute the BloomLSH signature for a given text."""
        shingles = self._get_shingles(text)
        signature = self._get_minhash_signature(shingles)
        bands = self._get_band_signature(signature)
        return bands.astype(np.uint32)


class BandedBloomFilter:
    """Implementation of a bloom filter specifically for BloomLSH."""

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
        self.lsh_permutations = lsh_permutations
        self.lsh_threshold = lsh_threshold
        self.lsh_ngram_size = lsh_ngram_size
        self.bloom_size = bloom_size
        self.bloom_hashes = bloom_hashes
        self.num_bands, self.band_size = self._set_bands()
        self.signature = Signature(
            lsh_threshold=lsh_threshold,
            lsh_permutations=lsh_permutations,
            ngrams=lsh_ngram_size,
            num_bands=self.num_bands,
            band_size=self.band_size,
            seed=lsh_seed,
            prime=lsh_prime,
        )
        self.state = np.zeros((self.num_bands, bloom_size), dtype=bool)

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
        band_values = self._signature(data)
        for band_idx, value in enumerate(band_values):
            hash_indices = self._hashes(value)
            if np.all(self.state[band_idx, hash_indices]):
                return True
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
class DeduplicationFormatter(BaseModel):
    """Formatter to deduplicate the text of rows across the whole dataset on paragraph level.

    Uses a BloomLSH filter internally to detect similar content: https://arxiv.org/pdf/2411.04257v1
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(default="deduplication_formatter")
    on: str = Field(default="text")
    to: str = Field(default="text")
    bloom_size: int = Field(default=1_000_000, gt=0)
    bloom_hashes: int = Field(default=3, gt=0)
    lsh_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    lsh_permutations: int = Field(default=256, gt=0)
    ngrams: int = Field(default=8, gt=0)
    split_char: str | None = Field(default="\n")
    memory: int = Field(default=256, gt=0)
    use_distributed_actor: bool = Field(default=True)
    num_bands: int = Field(default=0)
    band_size: int = Field(default=0)

    _bloom: Optional[BandedBloomFilter] = PrivateAttr(default=None)
    _actor: Any = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        bloom = BandedBloomFilter(
            bloom_size=self.bloom_size,
            bloom_hashes=self.bloom_hashes,
            lsh_permutations=self.lsh_permutations,
            lsh_threshold=self.lsh_threshold,
            lsh_ngram_size=self.ngrams,
        )
        object.__setattr__(self, "num_bands", bloom.num_bands)
        object.__setattr__(self, "band_size", bloom.band_size)

        if self.use_distributed_actor:
            object.__setattr__(
                self,
                "_actor",
                _get_or_create_actor(
                    name="bloom_actor",
                    memory=self.memory * 1024 * 1024,
                    bloom_size=self.bloom_size,
                    bloom_hashes=self.bloom_hashes,
                    lsh_permutations=self.lsh_permutations,
                    lsh_threshold=self.lsh_threshold,
                    lsh_ngram_size=self.ngrams,
                ),
            )
        else:
            object.__setattr__(self, "_bloom", bloom)

    def __call__(self, row: Row) -> Row:
        """Single formatting step.

        Decompose the document into paragraphs. For each paragraph, compute a signature and check if it
        matches the bloom filter. If not, insert it and continue. Otherwise, mark as duplicate and continue.
        Finally, re-assemble the document text only with the non-duplicate parts.
        """
        text = get_field(row, self.on)
        if not text:
            return row
        paragraphs = text.split(self.split_char) if self.split_char is not None else [text]
        unique = []
        for paragraph in paragraphs:
            if self.use_distributed_actor:
                match = ray.get(self._actor.get.remote(paragraph))
            else:
                match = self._bloom.get(paragraph)
            if not match:
                if self.use_distributed_actor:
                    ray.get(self._actor.put.remote(paragraph))
                else:
                    self._bloom.put(paragraph)
                unique.append(True)
            else:
                unique.append(False)
        if not any(unique):
            set_field(row, self.to, "")
        else:
            paragraphs = [p for (p, keep) in zip(paragraphs, unique, strict=False) if keep]
            set_field(row, self.to, self.split_char.join(paragraphs) if self.split_char is not None else paragraphs[0])
        return row


MapFn.register(DeduplicationFormatter)
