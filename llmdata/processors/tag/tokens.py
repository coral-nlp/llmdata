from typing import Any

from pydantic import Field, PrivateAttr

from llmdata.core.dependencies import requires
from llmdata.core.ops import MapFn, Row
from llmdata.core.registry import components
from llmdata.core.utils import get_field, set_field, silence


@requires("transformers")
@components.add("tag", "token_count")
class TokenCountTagger(MapFn):
    """Count tokens in text using a specified tokenizer.

    This processor uses HuggingFace tokenizers to count
    the number of tokens in text content.
    """

    # Override base fields with specific _defaults
    name: str = Field(default="token_count_tagger", description="Name of the token count tagger")
    on: str = Field(default="text", description="Column containing text to tokenize")
    to: str = Field(default="metadata.token_count", description="Column to write token count to")

    # Tokenizer configuration
    pretrained_tokenizer_name_or_path: str = Field(
        default="openai-community/gpt2",
        description="Name or path of the pretrained tokenizer to use",
    )
    add_special_tokens: bool = Field(default=False, description="Whether to add special tokens during tokenization")
    use_fast: bool = Field(default=True, description="Whether to use fast tokenizer implementation")

    # Private attributes (not part of the pydantic model validation)
    _tokenizer: Any = PrivateAttr()

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.pretrained_tokenizer_name_or_path, use_fast=self.use_fast)

    def __call__(self, row: Row) -> Row:
        """Calculate the token count of the given text."""
        text = get_field(row, self.on)
        with silence():  # the "this is longer than model max length" warning is annoying
            token_info = self._tokenizer(
                [text],  # Inefficient to not do it in batches, but this would require rework of MapFn
                return_length=True,
                padding=False,
                truncation=False,
                max_length=None,
                add_special_tokens=self.add_special_tokens,
            )
        num_tokens = token_info["length"][0]
        set_field(row, self.to, num_tokens if num_tokens else 0)
        return row


@components.add("tag", "length")
class LengthTagger(MapFn):
    """Character and word count tagger."""

    # Override base fields with specific _defaults
    name: str = Field(default="length_tagger", description="Name of the length tagger")
    on: str = Field(default="text", description="Column containing text to analyze")
    to: str = Field(default="metadata.length", description="Column to write length statistics to")

    # Length calculation options
    count_characters: bool = Field(default=True, description="Whether to count characters")
    count_words: bool = Field(default=True, description="Whether to count words")
    count_lines: bool = Field(default=True, description="Whether to count lines")
    count_paragraphs: bool = Field(default=False, description="Whether to count paragraphs")
    word_delimiter: str = Field(default=" ", description="Delimiter to use for word counting")

    def __call__(self, row: Row) -> Row:
        """Calculate length statistics for the given text."""
        text = get_field(row, self.on) or ""

        stats = {}

        if self.count_characters:
            stats["char_count"] = len(text)

        if self.count_words:
            words = text.split(self.word_delimiter) if text else []
            stats["word_count"] = len([w for w in words if w.strip()])

        if self.count_lines:
            stats["line_count"] = text.count("\n") + 1 if text else 0

        if self.count_paragraphs:
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            stats["paragraph_count"] = len(paragraphs)

        set_field(row, self.to, stats)
        return row
