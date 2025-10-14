import re
import string
from collections import Counter
from typing import Any, Literal

import numpy as np
from ocroscope import ocr_evaluation
from pydantic import Field

from llmdata.core.ops import MapFn, Row
from llmdata.core.registry import components
from llmdata.core.utils import get_field, set_field


@components.add("tag", "gopher_quality")
class GopherQualityTagger(MapFn):
    """Tagger that computes Gopher quality metrics.

    This processor computes various quality metrics used in the Gopher paper
    to assess text quality including word statistics, punctuation ratios,
    and structural features.
    """

    # Override base fields with specific _defaults
    name: str = Field(default="gopher_quality", description="Name of the Gopher quality tagger")
    on: str = Field(default="text", description="Column containing text to analyze")
    to: str = Field(default="metadata.gopher_quality", description="Column to write quality metrics to")

    # Quality analysis configuration
    language: Literal["en", "de"] = Field(default="de", description="Language for stop word detection and analysis")

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self._stop_words = {
            "en": {"the", "be", "to", "of", "and", "that", "have", "with"},
            "de": {
                "der",
                "die",
                "das",
                "den",
                "dem",
                "des",  # definite articles
                "ein",
                "eine",
                "einen",
                "einem",
                "einer",  # indefinite articles
                "und",
                "oder",
                "aber",  # conjunctions
                "ist",
                "sind",
                "hat",
                "haben",
                "wird",
                "werden",  # common verbs
                "von",
                "zu",
                "mit",
                "in",
                "auf",
                "für",
                "bei",
                "nach",
                "vor",
                "über",
                "unter",
                "durch",
                "gegen",
                "ohne",
                "um",  # prepositions
                "ich",
                "du",
                "er",
                "sie",
                "es",
                "wir",
                "ihr",
                "sich",
                "sein",
                "seine",
                "ihrer",
                "ihren",
                "mich",
                "dich",  # pronouns
                "nicht",
                "auch",
                "nur",
                "noch",
                "schon",  # adverbs
                "dass",
                "wenn",
                "als",
                "wie",  # subordinating conjunctions
                "an",
                "am",
                "im",
                "ins",
                "zum",
                "zur",
                "vom",
                "beim",  # contractions
                "was",
                "wer",
                "wo",
                "wann",
                "warum",
                "welche",
                "welcher",  # question words
                "alle",
                "viele",
                "einige",
                "andere",
                "jede",
                "jeden",
                "jeder",  # quantifiers
                "kann",
                "könnte",
                "muss",
                "soll",
                "will",
                "würde",  # modal verbs
                "hier",
                "dort",
                "da",
                "dann",
                "jetzt",
                "heute",  # temporal/spatial adverbs
                "sehr",
                "mehr",
                "weniger",
                "ganz",
                "gar",
                "etwa",  # degree adverbs
                "ja",
                "nein",
                "doch",
                "so",
                "also",
                "nun",
                "mal",  # particles and discourse markers
            },
        }[self.language]
        self._punctuation_set = set(string.punctuation)

    def __call__(self, row: Row) -> Row:
        """Calculate gopher quality metrics."""
        text = get_field(row, self.on)
        if not text:
            # Set default values for empty text
            set_field(
                row,
                self.to,
                {
                    "word_count": 0,
                    "avg_word_length": 0.0,
                    "hash_ratio": 0.0,
                    "ellipsis_ratio": 0.0,
                    "bullet_line_ratio": 0.0,
                    "ellipsis_line_ratio": 0.0,
                    "alpha_word_ratio": 0.0,
                    "stop_word_count": 0,
                    "overall_score": 0.0,
                },
            )
            return row

        words = " ".join(text.split(" ")).split(" ")  # Collapse multiple whitespace and split into words
        n_words = len(words)
        non_symbol_words = [w for w in words if any(ch not in self._punctuation_set for ch in w)]
        n_non_symbol_words = len(non_symbol_words)
        lines = text.splitlines()
        n_lines = len(lines)

        stats = {
            "word_count": n_non_symbol_words,
            "avg_word_length": float(np.mean([len(w) for w in non_symbol_words]) if non_symbol_words else 0.0),
            "hash_ratio": text.count("#") / max(n_words, 1),
            "ellipsis_ratio": (text.count("...") + text.count("…")) / max(n_words, 1),
            "bullet_line_ratio": (
                sum(s.lstrip().startswith("•") or s.lstrip().startswith("-") for s in lines) / n_lines
                if n_lines > 0
                else 0.0
            ),
            "ellipsis_line_ratio": (
                sum(s.rstrip().endswith("...") or s.rstrip().endswith("…") for s in lines) / n_lines
                if n_lines > 0
                else 0.0
            ),
            "stop_word_count": len(self._stop_words.intersection(set(words))),
            "alpha_word_ratio": (sum(any(c.isalpha() for c in w) for w in words) / n_words if n_words > 0 else 0.0),
        }
        set_field(row, self.to, stats)
        return row


@components.add("tag", "gopher_repetition")
class GopherRepetitionTagger(MapFn):
    """Tagger that computes Gopher repetition metrics.

    This processor analyzes text for various types of repetition including
    duplicate lines, paragraphs, and n-grams to assess content quality.
    """

    # Override base fields with specific _defaults
    name: str = Field(default="gopher_repetition", description="Name of the Gopher repetition tagger")
    on: str = Field(default="text", description="Column containing text to analyze")
    to: str = Field(default="metadata.gopher_repetition", description="Column to write repetition metrics to")

    # Repetition analysis configuration
    top_n_grams: tuple[int, ...] = Field(default=(2, 3, 4), description="N-gram sizes for top n-gram analysis")
    dup_n_grams: tuple[int, ...] = Field(
        default=(5, 6, 7, 8, 9, 10), description="N-gram sizes for duplicate n-gram analysis"
    )

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self._par_splitter = re.compile(r"\n{2,}")
        self._line_splitter = re.compile("\n+")

    @staticmethod
    def _get_n_grams(words: list[str], n: int) -> list[str]:
        return [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]

    @staticmethod
    def _find_duplicates(x: list[str]) -> tuple[int, int]:
        unique_x = set()
        duplicate_chars = 0
        duplicate_elements = 0
        for element in x:
            if element in unique_x:
                duplicate_chars += len(element)
                duplicate_elements += 1
            else:
                unique_x.add(element)
        return duplicate_elements, duplicate_chars

    @staticmethod
    def _find_top_duplicate(x: list[str]) -> int:
        if not x:
            return 0
        counter: Counter = Counter()
        for element in x:
            counter[element] += 1
        top_n_gram = counter.most_common(1)[0]
        return len(top_n_gram[0]) * top_n_gram[1]

    @staticmethod
    def _find_all_duplicate(words: list[str], n: int) -> int:
        n_words = len(words)
        unique = set()
        repeated_chars, idx = 0, 0
        while idx < n_words - n + 1:
            n_gram = "".join(words[idx : idx + n])
            if n_gram in unique:
                repeated_chars += len(n_gram)
                idx += n
            else:
                unique.add(n_gram)
                idx += 1
        return repeated_chars

    def __call__(self, row: Row) -> Row:
        """Calculate Gopher repetition metrics."""
        text = get_field(row, self.on)

        if not text:
            # Set default values for empty text
            repetition_data = {
                "dup_line_frac": 0.0,
                "dup_para_frac": 0.0,
                "dup_line_char_frac": 0.0,
                "dup_para_char_frac": 0.0,
                "overall_score": 1.0,  # No repetition = high score
            }

            # Add n-gram fields
            for n in self.top_n_grams:
                repetition_data[f"top_{n}_gram_char_frac"] = 0.0
            for n in self.dup_n_grams:
                repetition_data[f"dup_{n}_gram_char_frac"] = 0.0

            set_field(row, self.to, repetition_data)
            return row

        text_len = len(text)

        # Paragraph duplicates
        paragraphs = self._par_splitter.split(text.strip())
        if paragraphs:
            para_duplicates, para_char_duplicates = self._find_duplicates(paragraphs)
            dup_para_frac = para_duplicates / len(paragraphs)
            dup_para_char_frac = para_char_duplicates / max(text_len, 1)
        else:
            dup_para_frac = 0.0
            dup_para_char_frac = 0.0

        repetition_data = {
            "dup_para_frac": dup_para_frac,
            "dup_para_char_frac": dup_para_char_frac,
        }

        # Line duplicates
        lines = self._line_splitter.split(text)
        if lines:
            line_duplicates, line_char_duplicates = self._find_duplicates(lines)
            dup_line_frac = line_duplicates / len(lines)
            dup_line_char_frac = line_char_duplicates / max(text_len, 1)
        else:
            dup_line_frac = 0.0
            dup_line_char_frac = 0.0

        repetition_data["dup_line_frac"] = dup_line_frac
        repetition_data["dup_line_char_frac"] = dup_line_char_frac

        words = " ".join(text.split(" ")).split(" ")  # collapse whitespace and separate into words

        # Top n-gram analysis
        for n in self.top_n_grams:
            n_grams = self._get_n_grams(words, n)
            if n_grams:
                top_char_length = self._find_top_duplicate(n_grams)
                top_char_frac = top_char_length / max(text_len, 1)
            else:
                top_char_frac = 0.0
            repetition_data[f"top_{n}_gram_char_frac"] = top_char_frac

        # Duplicate n-gram analysis
        for n in self.dup_n_grams:
            n_duplicates_char = self._find_all_duplicate(words, n)
            dup_char_frac = n_duplicates_char / max(text_len, 1)
            repetition_data[f"dup_{n}_gram_char_frac"] = dup_char_frac

        set_field(row, self.to, repetition_data)
        return row


@components.add("tag", "ocr_quality")
class OCRQualityTagger(MapFn):
    """Tagger that computes OCR-specific quality metrics.

    This processor analyzes text for common OCR errors and artifacts
    to assess the quality of OCR-processed text including character
    confusion patterns, spacing anomalies, and structural artifacts.
    """

    # Override base fields with specific defaults
    name: str = Field(default="ocr_quality", description="Name of the OCR quality tagger")
    on: str = Field(default="text", description="Column containing text to analyze")
    to: str = Field(default="metadata.ocr_quality", description="Column to write OCR quality metrics to")

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        # Spacing and word patterns
        self._spacing_patterns = {
            "missing_spaces": r"[a-z][A-Z]|[a-zA-Z]\d|\d[a-zA-Z]",
            "excessive_spaces": r" {3,}",
            "spaced_words": r"\b[a-zA-Z] [a-zA-Z] [a-zA-Z]\b",
            "very_long_words": r"\b\w{25,}\b",
        }
        self._spacing_patterns = {k: re.compile(v) for k, v in self._spacing_patterns.items()}  # type: ignore[misc]

        # Case anomaly patterns
        self._case_patterns = {
            "random_caps": r"\b[a-z]+[A-Z][a-z]*\b",
            "mixed_case_words": r"\b[a-zA-Z]*[a-z][A-Z][a-zA-Z]*\b",
        }
        self._case_patterns = {k: re.compile(v) for k, v in self._case_patterns.items()}  # type: ignore[misc]

        # Repeated character patterns
        self._repeat_patterns = {
            "repeated_chars": r"(.)\1{3,}",
            "repeated_sequences": r"(.{2,5})\1{2,}",
        }
        self._repeat_patterns = {k: re.compile(v) for k, v in self._repeat_patterns.items()}  # type: ignore[misc]

    def _calculate_spacing_anomaly_ratio(self, text: str) -> float:
        """Calculate ratio of spacing anomalies."""
        if not text:
            return 0.0

        words = text.split()
        if not words:
            return 0.0

        anomalies = 0
        anomalies += len(self._spacing_patterns["missing_spaces"].findall(text))  # type: ignore[attr-defined]
        anomalies += len(self._spacing_patterns["excessive_spaces"].findall(text))  # type: ignore[attr-defined]
        anomalies += len(self._spacing_patterns["spaced_words"].findall(text))  # type: ignore[attr-defined]
        anomalies += len(self._spacing_patterns["very_long_words"].findall(text))  # type: ignore[attr-defined]
        return min(anomalies / max(len(words), 1), 1.0)

    def _calculate_case_anomaly_ratio(self, text: str) -> float:
        """Calculate ratio of case anomalies."""
        if not text:
            return 0.0

        words = [w for w in text.split() if w.isalpha()]
        if not words:
            return 0.0

        anomalies = 0
        anomalies += len(self._case_patterns["random_caps"].findall(text))  # type: ignore[attr-defined]
        anomalies += len(self._case_patterns["mixed_case_words"].findall(text))  # type: ignore[attr-defined]
        return min(anomalies / max(len(words), 1), 1.0)

    def _calculate_word_fragment_ratio(self, text: str) -> float:
        """Calculate ratio of likely word fragments."""
        if not text:
            return 0.0

        words = text.split()
        if not words:
            return 0.0

        # Count very short words that are likely fragments
        fragments = 0
        for word in words:
            clean_word = re.sub(r"[^\w]", "", word)
            if (
                len(clean_word) == 1
                and clean_word.isalpha()
                or len(clean_word) == 2
                and clean_word.isalpha()
                and clean_word.lower()
                not in {
                    # English 2-letter words
                    "am",
                    "an",
                    "as",
                    "at",
                    "be",
                    "by",
                    "do",
                    "go",
                    "he",
                    "if",
                    "in",
                    "is",
                    "it",
                    "me",
                    "my",
                    "no",
                    "of",
                    "on",
                    "or",
                    "so",
                    "to",
                    "up",
                    "us",
                    "we",
                    # German 2-letter words
                    "ab",
                    "ad",
                    "au",
                    "da",
                    "du",
                    "eh",
                    "ei",
                    "er",
                    "es",
                    "ex",
                    "im",
                    "ja",
                    "je",
                    "la",
                    "ob",
                    "oh",
                    "um",
                    "wo",
                    "zu",
                }
            ):
                fragments += 1

        return min(fragments / max(len(words), 1), 1.0)

    def _calculate_line_artifact_ratio(self, text: str) -> float:
        """Calculate ratio of lines that are likely artifacts."""
        if not text:
            return 0.0

        lines = text.splitlines()
        if not lines:
            return 0.0

        artifacts = 0
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Single characters or very short lines
            if (
                len(line) <= 2
                or re.match(r"^[^\w\s]+$", line)
                or re.match(r"^\d+$", line)
                or re.match(r"^\d+\s*$|^[IVX]+\s*$|^Page\s+\d+", line, re.IGNORECASE)
            ):
                artifacts += 1

        return min(artifacts / max(len(lines), 1), 1.0)

    def _calculate_special_char_density(self, text: str) -> float:
        """Calculate density of unusual special characters."""
        if not text:
            return 0.0

        # Count unusual characters that often result from OCR errors
        unusual_chars = 0
        for char in text:
            # Various unicode ranges that often appear in broken OCR
            if (
                char in "«»" "''‚„‹›¡¿¦§¨©ª«¬®¯°±²³´µ¶·¸¹º»¼½¾¿"  # noqa: RUF001
                or char in "†‡•…‰‹›€™"  # Special punctuation # noqa: RUF001
                or 0x2000 <= ord(char) <= 0x206F  # General punctuation block
                or 0x2700 <= ord(char) <= 0x27BF  # Dingbats block
            ):
                unusual_chars += 1

        return min(unusual_chars / max(len(text), 1), 1.0)

    def _calculate_repeated_char_ratio(self, text: str) -> float:
        """Calculate ratio of repeated character sequences."""
        if not text:
            return 0.0

        repeated_chars = 0

        # Find sequences of repeated characters (3+ repetitions)
        for match in self._repeat_patterns["repeated_chars"].finditer(text):  # type: ignore[attr-defined]
            repeated_chars += len(match.group(0))

        # Find repeated sequences
        for match in self._repeat_patterns["repeated_sequences"].finditer(text):  # type: ignore[attr-defined]
            repeated_chars += len(match.group(0))

        return min(repeated_chars / max(len(text), 1), 1.0)

    def _calculate_numeric_context_errors(self, text: str) -> float:
        """Calculate ratio of numbers appearing in inappropriate word contexts."""
        words = text.split()
        errors = 0
        for word in words:
            # Numbers embedded in words (excluding common patterns like "2nd", "1st")
            if (
                re.search(r"[a-zA-Z]\d|d[a-zA-Z]", word)
                and not re.match(r"^\d+(st|nd|rd|th)$", word, re.IGNORECASE)
                or re.search(r"\d[a-zA-Z][a-zA-Z]*\d", word)
            ):
                errors += 1

        return min(errors / max(len(words), 1), 1.0)

    def _calculate_length_distribution(self, text: str) -> tuple[float, float]:
        """Calculate word length distribution (mean/std)."""
        words = [re.sub(r"[^\w]", "", word) for word in text.split() if re.sub(r"[^\w]", "", word)]
        word_lengths = [len(word) for word in words]
        avg_length = np.mean(word_lengths)
        std_length = np.std(word_lengths) if len(word_lengths) > 1 else 0.0

        return float(avg_length), float(std_length)

    def _calculate_length_extremes(self, text: str) -> tuple[float, float]:
        """Calculate ratio of very short and very long words."""
        words = [re.sub(r"[^\w]", "", word) for word in text.split() if re.sub(r"[^\w]", "", word)]
        if not words:
            return 0.0, 0.0

        word_lengths = [len(word) for word in words]
        very_short = sum(1 for length in word_lengths if length <= 1) / max(len(words), 1)
        very_long = sum(1 for length in word_lengths if length >= 15) / max(len(words), 1)

        return very_short, very_long

    def __call__(self, row: Row) -> Row:
        """Calculate OCR quality metrics."""
        text = get_field(row, self.on)

        if not text or not text.strip():
            # Set default values for empty text
            ocr_stats = {
                "spacing_anomaly_ratio": 0.0,
                "case_anomaly_ratio": 0.0,
                "word_fragment_ratio": 0.0,
                "line_artifact_ratio": 0.0,
                "special_char_density": 0.0,
                "repeated_char_ratio": 0.0,
                "numeric_context_errors": 0.0,
                "word_length_avg": 0.0,
                "word_length_std": 0.0,
                "ratio_very_short_words": 0.0,
                "ratio_very_long_words": 0.0,
            }
            set_field(row, self.to, ocr_stats)
            return row

        avg_length, std_length = self._calculate_length_distribution(text)
        ratio_short, ratio_long = self._calculate_length_extremes(text)

        ocr_stats = {
            "spacing_anomaly_ratio": self._calculate_spacing_anomaly_ratio(text) or 0.0,
            "case_anomaly_ratio": self._calculate_case_anomaly_ratio(text) or 0.0,
            "word_fragment_ratio": self._calculate_word_fragment_ratio(text) or 0.0,
            "line_artifact_ratio": self._calculate_line_artifact_ratio(text) or 0.0,
            "special_char_density": self._calculate_special_char_density(text) or 0.0,
            "repeated_char_ratio": self._calculate_repeated_char_ratio(text) or 0.0,
            "numeric_context_errors": self._calculate_numeric_context_errors(text) or 0.0,
            "word_length_avg": avg_length or 0.0,
            "word_length_std": std_length or 0.0,
            "ratio_very_short_words": ratio_short or 0.0,
            "ratio_very_long_words": ratio_long or 0.0,
        }

        set_field(row, self.to, ocr_stats)
        return row


@components.add("tag", "ocroscope")
class OCRoscopeTagger(MapFn):
    """Tagger that applies OCRoscope to text."""

    # Override base fields with specific defaults
    name: str = Field(default="ocroscope_quality", description="Name of the OCR quality tagger")
    on: str = Field(default="text", description="Column containing text to analyze")
    to: str = Field(default="ocr_quality", description="Column to write OCR quality metrics to")
    max_chars: int = Field(default=2**16, description="Maximum number of characters to use for perplexity estimation")

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)

    def __call__(self, row: Row) -> Row:
        """Calculate OCR quality metrics."""
        text = get_field(row, self.on) or ""
        text = text.encode("utf-8", "ignore").decode("utf-8", "replace")
        # Remove control characters except common whitespace
        text = "".join(c for c in text if ord(c) >= 32 or c in "\t\n")
        try:
            if len(text) > self.max_chars:
                text = text[: self.max_chars]
            ocr_estimate = ocr_evaluation(id=None, text=text)
            ocr_estimate.calculate_ocr_rate()
            score = ocr_estimate.ratio_segment
        except Exception:
            score = -1
        set_field(row, self.to, score if score is not None else -1)
        return row
