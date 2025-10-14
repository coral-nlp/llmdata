from typing import Any

from pydantic import Field

from llmdata.core.dependencies import requires
from llmdata.core.ops import MapFn, Row
from llmdata.core.registry import components
from llmdata.core.utils import get_field, set_field


@requires("fasttext")
@components.add("tag", "language")
class LanguageTagger(MapFn):
    """Tag text with language detection using FastText.

    This processor uses FastText's language identification model to detect
    the language of text content and add language metadata.
    """

    # Override default fields with specific documentation
    name: str = Field(default="language_tagger", description="Name of the language tagger")
    on: str = Field(default="text", description="Column containing text to analyze for language")
    to: str = Field(default="metadata.language", description="Column to write language detection results to")

    # Language detection specific fields
    fasttext_model_path: str | None = Field(
        default=None, description="Path to custom FastText language model (uses default if None)"
    )
    k: int = Field(default=1, description="Number of top language predictions to return", ge=1, le=10)
    max_tokens: int = Field(
        default=4096,
        description="Maximum number of characters to analyze for language detection",
        gt=0,
    )
    confidence_threshold: float = Field(
        default=0.0, description="Minimum confidence score for language predictions", ge=0.0, le=1.0
    )

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self._model = None

    @property
    def model(self):  # type: ignore
        """The underlying FastText detection model."""
        if self._model is None:
            from urllib.request import urlretrieve

            from fasttext.FastText import _FastText

            model_path, _ = urlretrieve("https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin")  # nosec
            self._model = _FastText(model_path)
        return self._model

    def __call__(self, row: Row) -> Row:
        """Add language detection to metadata."""
        text = get_field(row, self.on)
        if not text or not text.strip():
            set_field(
                row,
                self.to,
                {
                    "names": ["unknown"],
                    "scores": [0.0],
                },
            )
            return row

        # FastText expects single line, replace newlines with spaces
        text_for_detection = text.replace("\n", " ").strip()

        # Truncate if needed
        if len(text_for_detection) > self.max_tokens:
            text_for_detection = text_for_detection[: self.max_tokens]

        try:
            predictions = self.model.predict(text_for_detection, k=self.k)

            # Extract language codes and scores
            languages: list = [pred.replace("__label__", "") for pred in predictions[0]]
            scores: list = predictions[1].tolist()

            # Filter by confidence lsh_threshold
            if self.confidence_threshold > 0:
                filtered_results = [
                    (lang, score)
                    for lang, score in zip(languages, scores, strict=False)
                    if score >= self.confidence_threshold
                ]
                if filtered_results:
                    languages, scores = zip(*filtered_results, strict=False)  # type: ignore[assignment]
                    languages, scores = list(languages), list(scores)
                else:
                    languages, scores = ["unknown"], [0.0]

            set_field(
                row,
                self.to,
                {
                    "names": languages,
                    "scores": scores,
                },
            )
        except Exception:
            # Fallback on error
            set_field(
                row,
                self.to,
                {
                    "names": ["unknown"],
                    "scores": [0.0],
                },
            )

        return row
