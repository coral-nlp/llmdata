import re
import unicodedata
import urllib.request
from typing import Any, ClassVar, Literal

import kenlm
import sentencepiece
from pydantic import Field

from llmdata.core.ops import MapFn, Row
from llmdata.core.registry import components
from llmdata.core.utils import get_field, set_field

URLS = {
    "kenlm_model": "https://huggingface.co/ocisd4/kenlm/resolve/main/wikipedia/{language}.arpa.bin",
    "sp_vocab": "https://huggingface.co/ocisd4/kenlm/resolve/main/wikipedia/{language}.sp.vocab",
    "sp_model": "https://huggingface.co/ocisd4/kenlm/resolve/main/wikipedia/{language}.sp.model",
}


def download_to_tmp(url: str) -> str:
    """Download the KenLM model files to temporary file."""
    path, _ = urllib.request.urlretrieve(url)  # nosec B310
    return path


class SentencePiece:
    """SentencePiece tokenizer."""

    def __init__(
        self,
        model: str,
    ):
        super().__init__()
        self.sp = sentencepiece.SentencePieceProcessor()
        self.sp.load(str(model))

    def __call__(self, text: str) -> str:
        """Tokenize a piece of text."""
        tokenized = self.sp.encode_as_pieces(text)
        return " ".join(tokenized)


class KenlmModel:
    """Taken from https://huggingface.co/ocisd4/kenlm/blob/main/model.py for compatibility."""

    digit_re: re.Pattern = re.compile(r"\d")
    unicode_punct: ClassVar[dict[str, str]] = {
        "，": ",",  # noqa: RUF001
        "。": ".",
        "、": ",",
        "„": '"',
        "”": '"',
        "“": '"',
        "«": '"',
        "»": '"',
        "１": '"',  # noqa: RUF001
        "」": '"',
        "「": '"',
        "《": '"',
        "》": '"',
        "´": "'",  # noqa: RUF001
        "∶": ":",  # noqa: RUF001
        "：": ":",  # noqa: RUF001
        "？": "?",  # noqa: RUF001
        "！": "!",  # noqa: RUF001
        "（": "(",  # noqa: RUF001
        "）": ")",  # noqa: RUF001
        "；": ";",  # noqa: RUF001
        "–": "-",  # noqa: RUF001
        "—": " - ",
        "．": ". ",  # noqa: RUF001
        "～": "~",  # noqa: RUF001
        "’": "'",  # noqa: RUF001
        "…": "...",
        "━": "-",
        "〈": "<",
        "〉": ">",
        "【": "[",
        "】": "]",
        "％": "%",  # noqa: RUF001
        "►": "-",
    }
    unicode_punct_re = re.compile(f"[{''.join(unicode_punct.keys())}]")
    non_printing_chars_re = re.compile(f"[{''.join(map(chr, list(range(0,32)) + list(range(127,160))))}]")
    kenlm_model_dir = None
    sentence_piece_model_dir = None

    def __init__(
        self,
        language: Literal["en", "de"],
        lower_case: bool = False,
        remove_accents: bool = False,
        normalize_numbers: bool = True,
        punctuation: int = 1,
    ):
        self.model = kenlm.Model(download_to_tmp(URLS["kenlm_model"].format(language=language)))
        self.tokenizer = SentencePiece(download_to_tmp(URLS["sp_model"].format(language=language)))
        self.accent = remove_accents
        self.case = lower_case
        self.numbers = normalize_numbers
        self.punct = punctuation

    @classmethod
    def from_pretrained(
        cls,
        language: Literal["en", "de"],
    ) -> "KenlmModel":
        """Instantiate a Kenlm model from a pre-trained model."""
        return cls(
            language,
            False,
            False,
            True,
            1,
        )

    def _pp(self, log_score: float, length: float) -> float:
        return float(10.0 ** (-log_score / length))

    def get_perplexity(self, doc: str, normalize_cc_net: bool = True) -> float:
        """Calculate perplexity score for a given text."""
        if normalize_cc_net:
            doc = self.normalize(
                doc,
                accent=self.accent,
                case=self.case,
                numbers=self.numbers,
                punct=self.punct,
            )
        # Tokenize (after normalizing): See https://github.com/facebookresearch/cc_net/blob/bda555bd1cf1ee2e0b925363e62a61cd46c8b60d/cc_net/mine.py#L352 for full pipeline
        doc = self.tokenizer(doc)
        doc_log_score: float = 0.0
        doc_length: float = 0.0
        for line in doc.split("\n"):
            log_score = self.model.score(line)
            length = len(line.split()) + 1
            doc_log_score += float(log_score)
            doc_length += float(length)
        return round(self._pp(doc_log_score, doc_length), 1)

    def normalize(
        self,
        line: str,
        accent: bool = True,
        case: bool = True,
        numbers: bool = True,
        punct: int = 1,
    ) -> str:
        """Normalize a line of text."""
        line = line.strip()
        if not line:
            return line
        if case:
            line = line.lower()
        if accent:
            line = self.strip_accents(line)
        if numbers:
            line = self.digit_re.sub("0", line)
        if punct == 1:
            line = self.replace_unicode_punct(line)
        elif punct == 2:
            line = self.remove_unicode_punct(line)
        line = self.remove_non_printing_char(line)
        return line

    def strip_accents(self, line: str) -> str:
        """Strip accents from a piece of text."""
        nfd = unicodedata.normalize("NFD", line)
        output = [c for c in nfd if unicodedata.category(c) != "Mn"]
        if len(output) == line:
            return line
        return "".join(output)

    def replace_unicode_punct(self, text: str) -> str:
        """Replace unicode punctuation with accents."""
        return "".join(self.unicode_punct.get(c, c) for c in text)

    def remove_unicode_punct(self, text: str) -> str:
        """More aggressive version of replace_unicode_punct but also faster."""
        return self.unicode_punct_re.sub("", text)

    def remove_non_printing_char(self, text: str) -> str:
        """Remove non-printing chars from text."""
        return self.non_printing_chars_re.sub("", text)


@components.add("tag", "perplexity")
class PerplexityTagger(MapFn):
    """Tagger that applies a KenLM model to text to calculate perplexity."""

    # Override base fields with specific defaults
    name: str = Field(default="ocr_quality", description="Name of the perplexity tagger")
    on: str = Field(default="text", description="Column containing text to analyze")
    to: str = Field(default="perplexity", description="Column to write perplexity score to")
    language: Literal["de", "en"] = Field(default="de", description="Language to estimate perplexity for")
    max_chars: int = Field(default=2**16, description="Maximum number of characters to use for perplexity estimation")

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self._model = None

    @property
    def model(self):  # type: ignore
        """The underlying KenLM detection model."""
        if self._model is None:
            self._model = KenlmModel.from_pretrained(self.language)
        return self._model

    def __call__(self, row: Row) -> Row:
        """Add perplexity to metadata."""
        text = get_field(row, self.on) or ""
        text = text.encode("utf-8", "ignore").decode("utf-8", "replace")
        # Remove control characters except common whitespace
        text = "".join(c for c in text if ord(c) >= 32 or c in "\t\n")
        try:
            if len(text) > self.max_chars:
                score = self.model.get_perplexity(text[: self.max_chars])
            else:
                score = self.model.get_perplexity(text)
        except Exception:
            score = -1
        set_field(row, self.to, score)
        return row
