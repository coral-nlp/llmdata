import re
import string
from collections import Counter
from typing import Any

from pydantic import Field

from llmdata.core.ops import MapFn, Row
from llmdata.core.registry import components
from llmdata.core.utils import get_field, set_field


@components.add("tag", "ngrams")
class NgramsCountTagger(MapFn):
    """Extract ngram counts from samples."""

    name: str = Field(default="ngram_tagger", description="Name of the ngram tagger")
    on: str = Field(default="text", description="Column containing text to tag with ngram counts")
    to: str = Field(default="metadata.ngrams", description="Column to write ngram counts to")
    ngram_size: int = Field(default=5, description="The gram sizes to count")
    top_k: int = Field(default=None, description="The top k ngram counts to keep")

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)

    def __call__(self, row: Row) -> Row:
        """Add ngram counts to metadata."""
        text = get_field(row, self.on)
        if not text or not text.strip():
            set_field(row, self.to, {})
            return row

        words = re.split(f"[{re.escape(string.punctuation + string.whitespace)}]+", text)
        ngrams = [" ".join(words[i : i + self.ngram_size]) for i in range(len(words) - self.ngram_size + 1)]
        stats = Counter(ngrams).most_common(self.top_k)

        set_field(row, self.to, stats)
        return row
