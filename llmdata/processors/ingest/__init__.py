import uuid
from typing import Any, List

from pydantic import Field

from llmdata.core.ops import MapFn, Row
from llmdata.core.registry import components
from llmdata.core.utils import get_field, set_field


@components.add("ingest", "base")
class BaseIngestor(MapFn):
    """Ingestor to conform arbitrary column layouts into llmdata format."""

    # Override default fields with more specific documentation
    name: str = Field(default="plain_ingest", description="Name of the extractor")
    on: str = Field(default=None, description="Not used")
    to: str = Field(default=None, description="Not used")
    id_column: str = Field(
        description="Column name in data to read identifier from; if not existent, inserts random UUIDs"
    )
    text_column: str = Field(
        description="Column name in data to read raw text from; if not existent, insert empty string"
    )
    source_name_or_column: str = Field(
        description="Column name in data to read source information from; if not existent, use this name as license instead"
    )
    subset_name_or_column: str | None = Field(
        default=None,
        description="Column name in data to read subset information from; if not existent, use this name as subset instead",
    )
    license_name_or_column: str | None = Field(
        default=None,
        description="Column name in data to read license from; if not existent, use this name as license instead",
    )
    other: list[str] | None = Field(default=None, description="Other columns to include")

    def __call__(self, row: Row) -> Row:
        """Ingests data to llmdata schema format."""
        # Test if license is column or name
        updated_row: dict[str, Any] = {
            "id": str(get_field(row, self.id_column)) or str(uuid.uuid4().hex),
            "text": get_field(row, self.text_column) or "",
            "source": get_field(row, self.source_name_or_column) or self.source_name_or_column,
            "metadata": {},
        }
        if self.other:  # Do it first in case it's the metadata column
            for col in self.other:
                updated_row[col] = get_field(row, col)
        if self.subset_name_or_column:
            updated_row["metadata"]["subset"] = (  # type: ignore[index]
                get_field(row, self.subset_name_or_column) or self.subset_name_or_column
            )
        if self.license_name_or_column:
            updated_row["metadata"]["license"] = (  # type: ignore[index]
                get_field(row, self.license_name_or_column) or self.license_name_or_column
            )

        return updated_row


__all__ = ["BaseIngestor"]
