from pydantic import Field

from llmdata.core.ops import FilterFn, Row
from llmdata.core.registry import components
from llmdata.core.utils import get_field


@components.add("filter", "language")
class LanguageFilter(FilterFn):
    """Filter text based on detected language.

    This filter keeps or removes content based on the language detection
    results, allowing filtering by language codes and confidence scores.
    """

    # Override base fields with specific _defaults
    name: str = Field(default="language_filter", description="Name of the language filter")
    on: str = Field(default="metadata.language", description="Column containing language detection results")
    if_missing: bool = Field(default=False, description="Return value when language data is missing")

    # Language filtering specific fields
    allowed_languages: str | list[str] | set[str] = Field(
        default="en", description="Language codes to allow (single string or list of strings)"
    )
    min_confidence: float = Field(
        default=0.5,
        description="Minimum confidence lsh_threshold for language detection",
        ge=0.0,
        le=1.0,
    )
    allow_partial_match: bool = Field(
        default=True,
        description="Whether to allow partial matches (True) or require all detected languages to be allowed (False)",
    )

    def __call__(self, row: Row) -> bool:
        """Check if the language and score match the given ones."""
        langs = get_field(row, self.on) or {}

        names = langs.get("names")
        scores = langs.get("scores")
        if not names or not scores:
            return self.if_missing

        if isinstance(names, str):
            names = [names]
        if isinstance(scores, float):
            scores = [scores]

        matches = [
            lang in self.allowed_languages and score >= self.min_confidence
            for lang, score in zip(names, scores, strict=False)
        ]
        if self.allow_partial_match:
            return any(matches)
        else:
            return all(matches)
