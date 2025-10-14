import re
from typing import Any

from pydantic import Field

from llmdata.core.dependencies import requires
from llmdata.core.ops import MapFn, Row
from llmdata.core.registry import components
from llmdata.core.utils import get_field, set_field


@requires("ftfy")
@components.add("format", "ftfy")
class FTFYFormatter(MapFn):
    """Formatter that uses ftfy to fix text encoding issues.

    ftfy (fix text for you) fixes Unicode that's been broken by encoding/decoding
    issues, mojibake, and other common text corruption problems.
    """

    name: str = Field(default="ftfy_formatter", description="Name of the formatter")
    on: str = Field(default="text", description="Column to read text from")
    to: str = Field(default="text", description="Column to save processed text to")
    fix_encoding: bool = Field(default=True, description="Fix encoding issues")
    fix_entities: bool = Field(default=True, description="Fix HTML entities")
    remove_terminal_escapes: bool = Field(default=True, description="Remove terminal escape sequences")
    fix_latin_ligatures: bool = Field(default=True, description="Fix Latin ligatures")
    fix_character_width: bool = Field(default=True, description="Fix character width issues")
    uncurl_quotes: bool = Field(default=True, description="Convert curly quotes to straight quotes")
    fix_line_breaks: bool = Field(default=True, description="Normalize line breaks")
    fix_surrogates: bool = Field(default=True, description="Fix surrogate pairs")
    remove_control_chars: bool = Field(default=True, description="Remove control characters")
    normalization: str = Field(default="NFC", description="Unicode normalization form (NFC, NFKC, etc.)")

    @property
    def ftfy_config(self) -> dict[str, Any]:
        """Return the FTFY config dict."""
        return {
            "fix_encoding": self.fix_encoding,
            "fix_entities": self.fix_entities,
            "remove_terminal_escapes": self.remove_terminal_escapes,
            "fix_latin_ligatures": self.fix_latin_ligatures,
            "fix_character_width": self.fix_character_width,
            "uncurl_quotes": self.uncurl_quotes,
            "fix_line_breaks": self.fix_line_breaks,
            "fix_surrogates": self.fix_surrogates,
            "remove_control_chars": self.remove_control_chars,
            "normalization": self.normalization,
        }

    def __call__(self, row: Row) -> Row:
        """Apply FTFY package for formatting to the given text."""
        import ftfy

        text = get_field(row, self.on)
        if not text:
            return row
        text = ftfy.fix_text(text, **self.ftfy_config)
        set_field(row, self.to, text)
        return row


@components.add("format", "spacing")
class SpaceFormatter(MapFn):
    """Formatter that fixes common spacing issues."""

    name: str = Field(default="ocr_error_formatter", description="Name of the formatter")
    on: str = Field(default="text", description="Column to read text from")
    to: str = Field(default="text", description="Column to save formatted text to")
    fix_umlaut_notation: bool = Field(default=True, description="Fix stray umlaut notation in OCR text")
    fix_hyphenation: bool = Field(default=True, description="Fix hyphenation at line breaks")
    normalize_whitespace: bool = Field(default=True, description="Normalize whitespace")
    normalize_line_breaks: bool = Field(default=True, description="Normalize line breaks")
    collapse_paragraph_breaks: bool = Field(default=True, description="Collapse in-paragraph line breaks")

    def __call__(self, row: Row) -> Row:
        """Apply spacing error correction to the given text."""
        text = get_field(row, self.on)
        if not text:
            return row

        # Normalize whitespace
        if self.normalize_whitespace:
            # Multiple spaces/tabs to single space
            text = re.sub(r"[ \t]+", " ", text)

        # Reduce multiple paragraph breaks
        if self.normalize_line_breaks:
            text = re.sub(r"\n\n\n+", "\n\n", text)

        # Fix hyphenation at line breaks (word- \n word -> wordword)
        if self.fix_hyphenation:
            text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)

        # Collapse in-paragraph line breaks
        if self.collapse_paragraph_breaks:
            text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

        # Clean up any extra spaces that might have been created
        text = re.sub(r"  +", " ", text)
        set_field(row, self.to, text)
        return row
