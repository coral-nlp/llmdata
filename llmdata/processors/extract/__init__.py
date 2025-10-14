from pydantic import Field

from llmdata.core.ops import MapFn, Row
from llmdata.core.registry import components
from llmdata.core.utils import get_field, set_field

from .html import HTMLExtractor
from .tei import TEIExtractor


@components.add("extract", "plain")
class PlainTextExtractor(MapFn):
    """Pass-through extractor for plain text."""

    # Override default fields with more specific documentation
    name: str = Field(default="plain_extractor", description="Name of the extractor")
    on: str = Field(default="text", description="Column containing text content to extract from")
    to: str = Field(default="text", description="Column to write extracted plain text to")

    def __call__(self, row: Row) -> Row:
        """Pass through plain text from/to specified fields."""
        text = get_field(row, self.on)
        set_field(row, self.to, text)
        return row


__all__ = ["HTMLExtractor", "PlainTextExtractor", "TEIExtractor"]
