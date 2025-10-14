from pydantic import Field

from llmdata.core import MapFn, Row, components, get_field, set_field


@components.add("extract", "html")
class HTMLExtractor(MapFn):
    """Extract text from HTML content.

    This processor extracts plain text from HTML content by removing
    HTML tags, scripts, and styles, leaving only the readable text.
    """

    # Override default fields with more specific documentation
    name: str = Field(default="html_extractor", description="Name of the HTML extractor")
    on: str = Field(default="text", description="Column containing HTML content to extract from")
    to: str = Field(default="text", description="Column to write extracted plain text to")

    # Additional configuration options
    favor_precision: bool = Field(default=True, description="Prefer less text but correct extraction.")
    timeout: bool = Field(default=True, description="The timeout for extraction, per document, in seconds")
    deduplicate: bool = Field(default=True, description="Apply trafilatura's deduplicate option")

    def __call__(self, row: Row) -> Row:
        """Parse plain text from HTML."""
        from trafilatura import extract

        text = get_field(row, self.on)
        text = extract(
            text,
            favor_precision=self.favour_precision,
            include_comments=False,
            deduplicate=self.deduplicate,
            **self.kwargs,
        )
        set_field(row, self.to, text)
        return row
