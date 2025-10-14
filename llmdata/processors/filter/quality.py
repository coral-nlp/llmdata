from typing import Literal

from pydantic import Field

from llmdata.core.ops import FilterFn, Row
from llmdata.core.registry import components
from llmdata.core.utils import get_field


@components.add("filter", "gopher_quality")
class GopherQualityFilter(FilterFn):
    """Filter text based on Gopher quality metrics.

    This filter implements quality checks inspired by the Gopher paper,
    filtering out text that doesn't meet certain quality thresholds.
    """

    # Override base fields
    name: str = Field(default="gopher_quality", description="Name of the quality filter")
    on: str = Field(default="metadata.gopher_quality", description="Column containing quality metrics")
    if_missing: bool = Field(default=True, description="Return value when quality metrics are missing")

    # Quality parameters
    min_avg_word_length: float = Field(default=4.8, description="Minimum average word length", gt=0)
    max_avg_word_length: float = Field(default=7.3, description="Maximum average word length", gt=0)
    max_symbol_word_ratio: float = Field(
        default=0.1, description="Maximum ratio of symbol characters to words", ge=0, le=1
    )
    max_bullet_line_ratio: float = Field(
        default=0.7, description="Maximum ratio of lines that start with bullet points", ge=0, le=1
    )
    max_ellipsis_line_ratio: float = Field(
        default=0.3, description="Maximum ratio of lines containing ellipsis", ge=0, le=1
    )
    max_non_alpha_words_ratio: float = Field(
        default=0.99, description="Maximum ratio of non-alphabetic words", ge=0, le=1
    )
    min_stop_words: int = Field(default=6, description="Minimum number of stop words required", ge=0)

    def __call__(self, row: Row) -> bool:
        """Check if the quality criteria conform to the set thresholds."""
        stats = get_field(row, self.on)
        if not stats:
            return self.if_missing

        return not (
            stats.get("stop_word_count", 100_000) < self.min_stop_words
            or stats.get("alpha_word_ratio", 0) > self.max_non_alpha_words_ratio
            or stats.get("ellipsis_line_ratio", 0) > self.max_ellipsis_line_ratio
            or stats.get("bullet_line_ratio", 0) > self.max_bullet_line_ratio
            or stats.get("avg_word_length", 0) > self.max_avg_word_length
            or stats.get("avg_word_length", 100) < self.min_avg_word_length
            or stats.get("ellipsis_ratio", 0) > self.max_symbol_word_ratio
            or stats.get("hash_ratio", 0) > self.max_symbol_word_ratio
        )


@components.add("filter", "gopher_repetition")
class GopherRepetitionFilter(FilterFn):
    """Filter text based on Gopher repetition metrics.

    This filter removes text with excessive repetition patterns including
    duplicate lines, paragraphs, and n-gram repetitions.
    """

    # Override base fields
    name: str = Field(default="gopher_repetition", description="Name of the repetition filter")
    on: str = Field(default="metadata.gopher_repetition", description="Column containing repetition metrics")
    if_missing: bool = Field(default=True, description="Return value when repetition metrics are missing")

    # Line and paragraph repetition thresholds
    max_dup_line_frac: float | None = Field(
        default=0.25, description="Maximum fraction of duplicate lines allowed", ge=0, le=1
    )
    max_dup_para_frac: float | None = Field(
        default=0.3, description="Maximum fraction of duplicate paragraphs allowed", ge=0, le=1
    )
    max_dup_line_char_frac: float | None = Field(
        default=0.15, description="Maximum fraction of characters in duplicate lines", ge=0, le=1
    )
    max_dup_para_char_frac: float | None = Field(
        default=0.2,
        description="Maximum fraction of characters in duplicate paragraphs",
        ge=0,
        le=1,
    )

    # Flexible n-gram repetition thresholds
    top_n_gram_thresholds: tuple[tuple[int, float], ...] = Field(
        default=((2, 0.07), (3, 0.10), (4, 0.13)),
        description="Top n-gram bloom_size and maximum character fraction thresholds",
    )
    dup_n_gram_thresholds: tuple[tuple[int, float], ...] = Field(
        default=((5, 0.39), (6, 0.39), (7, 0.38), (8, 0.38), (9, 0.37), (10, 0.37)),
        description="Duplicate n-gram bloom_size and maximum character fraction thresholds",
    )

    def __call__(self, row: Row) -> bool:
        """Check if the repetition criteria conform to the set thresholds."""
        stats = get_field(row, self.on)
        if not stats:
            return self.if_missing

        # Line and paragraph checks
        if self.max_dup_line_frac is not None and stats.get("dup_line_frac", 0) > self.max_dup_line_frac:
            return False
        if self.max_dup_line_char_frac is not None and stats.get("dup_line_char_frac", 0) > self.max_dup_line_char_frac:
            return False
        if self.max_dup_para_frac is not None and stats.get("dup_para_frac", 0) > self.max_dup_para_frac:
            return False
        if self.max_dup_para_char_frac is not None and stats.get("dup_para_char_frac", 0) > self.max_dup_para_char_frac:
            return False

        # Top n-gram checks
        top = all(stats.get(f"top_{n}_gram_char_frac", 0) > threshold for n, threshold in self.top_n_gram_thresholds)

        # Duplicate n-gram checks
        dup = all(stats.get(f"dup_{n}_gram_char_frac", 0) <= threshold for n, threshold in self.dup_n_gram_thresholds)

        return top or dup


@components.add("filter", "ocr_quality")
class OCRQualityFilter(FilterFn):
    """Filter text based on OCR quality metrics.

    This filter removes text with poor OCR quality based on character
    confusion patterns, spacing anomalies, word fragments, and other
    OCR-specific quality indicators.
    """

    # Override base fields
    name: str = Field(default="ocr_quality", description="Name of the OCR quality filter")
    on: str = Field(default="metadata.ocr_quality", description="Column containing OCR quality metrics")
    if_missing: bool = Field(default=True, description="Return value when OCR quality metrics are missing")

    # Individual metric thresholds (maximum allowed values)
    max_spacing_anomaly_ratio: float = Field(
        default=0.15, description="Maximum spacing anomaly ratio allowed", ge=0, le=1
    )
    max_case_anomaly_ratio: float = Field(default=0.10, description="Maximum case anomaly ratio allowed", ge=0, le=1)
    max_word_fragment_ratio: float = Field(default=0.20, description="Maximum word fragment ratio allowed", ge=0, le=1)
    max_line_artifact_ratio: float = Field(default=0.25, description="Maximum line artifact ratio allowed", ge=0, le=1)
    max_special_char_density: float = Field(
        default=0.03, description="Maximum special character density allowed", ge=0, le=1
    )
    max_repeated_char_ratio: float = Field(
        default=0.05, description="Maximum repeated character ratio allowed", ge=0, le=1
    )
    max_numeric_context_errors: float = Field(
        default=0.08, description="Maximum numeric context error ratio allowed", ge=0, le=1
    )
    max_avg_length: float = Field(default=9, description="Maximum average word length allowed", ge=0)
    min_avg_length: float = Field(default=5, description="Minimum average word length allowed", ge=0)
    max_std_length: float = Field(default=5, description="Maximum word length std allowed", ge=0)
    min_std_length: float = Field(default=1, description="Minimum word length std allowed", ge=0)
    max_ratio_short: float = Field(default=0.1, description="Maximum ratio of short words", ge=0)
    max_ratio_long: float = Field(default=0.1, description="Maximum ratio of short words", ge=0)

    # Filter strictness
    filter_mode: Literal["any", "maj", "all"] = Field(
        default="strict",
        description="Filter strictness: 'any' (any thresholds exceeded), 'maj' (more thresholds exceeded than not), 'all' (all thresholds exceeded)",
    )

    def __call__(self, row: Row) -> bool:
        """Check if the OCR quality meets the specified thresholds."""
        stats = get_field(row, self.on)
        if not stats:
            return self.if_missing
        hits = (
            stats.get("spacing_anomaly_ratio", 0) > self.max_spacing_anomaly_ratio,
            stats.get("case_anomaly_ratio", 0) > self.max_case_anomaly_ratio,
            stats.get("word_fragment_ratio", 0) > self.max_word_fragment_ratio,
            stats.get("line_artifact_ratio", 0) > self.max_line_artifact_ratio,
            stats.get("special_char_density", 0) > self.max_special_char_density,
            stats.get("repeated_char_ratio", 0) > self.max_repeated_char_ratio,
            stats.get("numeric_context_errors", 0) > self.max_numeric_context_errors,
            stats.get("word_length_avg", 0) > self.max_avg_length,
            stats.get("word_length_avg", 0) < self.min_avg_length,
            stats.get("word_length_std", 0) > self.max_std_length,
            stats.get("word_length_std", 0) > self.min_std_length,
            stats.get("ratio_very_short_words", 0) > self.max_ratio_short,
            stats.get("ratio_very_long_words", 0) > self.max_ratio_long,
        )
        if self.filter_mode == "strict":
            return any(hits)
        if self.filter_mode == "lenient":
            return sum(hits) / len(hits) >= 0.5
        if self.filter_mode == "all":
            return all(hits)
        return self.if_missing
