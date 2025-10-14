from typing import Any

from pydantic import Field

from llmdata.core.ops import MapFn, Row
from llmdata.core.registry import components
from llmdata.core.utils import get_field, set_field

from .language import LanguageTagger
from .ngrams import NgramsCountTagger
from .perplexity import PerplexityTagger
from .quality import GopherQualityTagger, GopherRepetitionTagger, OCRoscopeTagger
from .tokens import LengthTagger, TokenCountTagger


@components.add("tag", "value")
class ValueTagger(MapFn):
    """Add a scalar value to the specified field.

    If value is a valid column name, copies its value over to the new field name.
    """

    on: str = Field(default=None, description="Input column to copy values from.")
    value: Any = Field(default=None, description="The value to insert.")
    replace_if_present: bool = Field(default=True, description="Whether to replace existing values")

    def model_post_init(self, *args: Any, **kwargs: dict[str, Any]) -> None:
        """Validate parameters."""
        if self.on is not None and self.value is not None:
            raise ValueError("Only one of `on` and `value` can be specified")

    def __call__(self, row: Row) -> Row:
        """Insert the specified value into the specified field."""
        existing_value = get_field(row, self.to)
        inserted_value = self.value or get_field(row, self.on)
        set_field(row, self.to, inserted_value if self.replace_if_present else existing_value)
        return row


__all__ = [
    "GopherQualityTagger",
    "GopherRepetitionTagger",
    "LanguageTagger",
    "LengthTagger",
    "TokenCountTagger",
    "PerplexityTagger",
    "OCRoscopeTagger",
    "ValueTagger",
]
