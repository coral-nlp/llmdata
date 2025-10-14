from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field, PrivateAttr

from llmdata.core.ops import MapFn, Row
from llmdata.core.registry import components
from llmdata.core.utils import get_field, set_field

if TYPE_CHECKING:
    from lxml.etree import Element  # nosec


class TEIParser:
    """Parser for TEI (Text Encoding Initiative) XML documents."""

    def __init__(self, output_format: Literal["markdown", "clean"] = "clean"):
        self.output_format = output_format.lower()
        self.namespaces = {
            "tei": "http://www.tei-c.org/ns/1.0",
            "xml": "http://www.w3.org/XML/1998/namespace",
        }

        # Elements to skip in clean format
        self.clean_skip_elements = {
            "note",
            "app",
            "rdg",
            "pb",
            "lb",
            "gap",
            "unclear",
            "supplied",
            "sic",
            "corr",
            "abbr",
            "expan",
            "bibl",
            "biblStruct",
            "ref",
            "ptr",
            "figure",
            "graphic",
            "figDesc",
            "fw",
            "signed",
            "dateline",
            "opener",
            "closer",
            "postscript",
            "trailer",
            "byline",
            "docAuthor",
            "docDate",
            "docTitle",
            "docImprint",
            "docEdition",
            "argument",
            "epigraph",
            "milestone",
            "anchor",
            "seg",
            "add",
            "del",
            "handShift",
            "space",
            "damage",
            "surplus",
            "subst",
            "mod",
            "undo",
            "redo",
        }

        # TEI element handlers
        self.handlers = {
            "text": self._handle_text,
            "body": self._handle_body,
            "div": self._handle_div,
            "p": self._handle_paragraph,
            "head": self._handle_heading,
            "title": self._handle_title,
            "hi": self._handle_highlight,
            "emph": self._handle_emphasis,
            "q": self._handle_quote,
            "quote": self._handle_quote,
            "ref": self._handle_reference,
            "ptr": self._handle_pointer,
            "list": self._handle_list,
            "item": self._handle_list_item,
            "note": self._handle_note,
            "app": self._handle_apparatus,
            "rdg": self._handle_reading,
            "lg": self._handle_line_group,
            "l": self._handle_line,
            "lb": self._handle_line_break,
            "pb": self._handle_page_break,
            "persName": self._handle_person_name,
            "placeName": self._handle_place_name,
            "orgName": self._handle_org_name,
            "date": self._handle_date,
            "name": self._handle_name,
            "table": self._handle_table,
            "row": self._handle_table_row,
            "cell": self._handle_table_cell,
            "bibl": self._handle_bibliography,
            "biblStruct": self._handle_bibl_struct,
            "author": self._handle_author,
            "editor": self._handle_editor,
            "pubPlace": self._handle_pub_place,
            "publisher": self._handle_publisher,
            "foreign": self._handle_foreign,
            "gap": self._handle_gap,
            "unclear": self._handle_unclear,
            "supplied": self._handle_supplied,
            "choice": self._handle_choice,
            "sic": self._handle_sic,
            "corr": self._handle_correction,
            "abbr": self._handle_abbreviation,
            "expan": self._handle_expansion,
        }

    def __call__(self, xml_content: str) -> str:
        """Remove TEI markup and extract plain text."""
        from defusedxml import ElementTree

        """Parse TEI XML and extract content."""
        root = ElementTree.fromstring(xml_content.encode("utf-8"))
        text_content = self._extract_text_content(root)
        root.clear()
        return text_content

    def _extract_text_content(self, root: "Element") -> str:
        """Extract and convert main text content."""
        text_elem = root.find(".//tei:text", self.namespaces)
        if text_elem is None:
            text_elem = root.find(".//text")
        if text_elem is None:
            text_elem = root.find(".//tei:body", self.namespaces)
            if text_elem is None:
                text_elem = root.find(".//body")

        if text_elem is not None:
            return self._process_element(text_elem)
        return self._process_element(root)

    def _process_element(self, element: "Element") -> str:
        from lxml.etree import QName  # nosec

        """Process an XML element and its children."""
        result = []
        try:
            tag = QName(element.tag).localname
            if self.output_format == "clean" and tag in self.clean_skip_elements:
                return ""

            if tag in self.handlers:
                return self.handlers[tag](element)

            if element.text:
                result.append(element.text)

        except:  # noqa
            pass  # nosec

        for child in element:
            child_content = self._process_element(child)
            if child_content:
                result.append(child_content)
            if child.tail:
                result.append(child.tail)

        return "".join(result)

    def _get_text_content(self, element: "Element") -> str:
        """Get all text content from element and descendants."""
        return "".join(element.itertext()).strip()

    def _process_children(self, element: "Element") -> str:
        """Process all children of an element."""
        result = []
        if element.text:
            result.append(element.text)
        for child in element:
            child_content = self._process_element(child)
            if child_content:
                result.append(child_content)
            if child.tail:
                result.append(child.tail)
        return "".join(result)

    # Element handlers (simplified versions)
    def _handle_text(self, element: "Element") -> str:
        return self._process_children(element)

    def _handle_body(self, element: "Element") -> str:
        return self._process_children(element)

    def _handle_div(self, element: "Element") -> str:
        content = self._process_children(element)
        return f"\n\n{content}\n\n" if content else ""

    def _handle_paragraph(self, element: "Element") -> str:
        content = self._process_children(element)
        return f"\n\n{content}\n\n" if content else ""

    def _handle_heading(self, element: "Element") -> str:
        content = self._get_text_content(element)
        if self.output_format == "clean":
            return f"\n\n{content}\n\n"
        return f"\n\n## {content}\n\n"

    def _handle_title(self, element: "Element") -> str:
        content = self._get_text_content(element)
        if self.output_format == "clean":
            return f"{content}\n\n"
        return f"# {content}\n\n"

    def _handle_highlight(self, element: "Element") -> str:
        content = self._process_children(element)
        if self.output_format == "clean":
            return content
        return f"**{content}**"

    def _handle_emphasis(self, element: "Element") -> str:
        content = self._process_children(element)
        if self.output_format == "clean":
            return content
        return f"*{content}*"

    def _handle_quote(self, element: "Element") -> str:
        content = self._process_children(element)
        if self.output_format == "clean":
            return content
        return f'"{content}"'

    def _handle_reference(self, element: "Element") -> str:
        if self.output_format == "clean":
            return ""
        return self._process_children(element)

    def _handle_pointer(self, element: "Element") -> str:
        return ""

    def _handle_list(self, element: "Element") -> str:
        if self.output_format == "clean":
            items = []
            for item in element.findall(".//item"):
                item_content = self._process_element(item).strip()
                if item_content:
                    items.append(item_content)
            return ". ".join(items) + ". " if items else ""
        return self._process_children(element)

    def _handle_list_item(self, element: "Element") -> str:
        content = self._process_children(element)
        if self.output_format == "clean":
            return content
        return f"- {content}\n"

    def _handle_note(self, element: "Element") -> str:
        return "" if self.output_format == "clean" else f" [{self._process_children(element)}] "

    def _handle_line_group(self, element: "Element") -> str:
        if self.output_format == "clean":
            lines = []
            for line in element.findall(".//l"):
                line_content = self._process_element(line).strip()
                if line_content:
                    lines.append(line_content)
            return " ".join(lines) + " " if lines else ""
        return self._process_children(element)

    def _handle_line(self, element: "Element") -> str:
        content = self._process_children(element)
        return content if self.output_format == "clean" else f"{content}\n"

    def _handle_line_break(self, element: "Element") -> str:
        return " " if self.output_format == "clean" else "\n"

    def _handle_page_break(self, element: "Element") -> str:
        return "" if self.output_format == "clean" else "\n\n"

    def _handle_person_name(self, element: "Element") -> str:
        return self._process_children(element)

    def _handle_place_name(self, element: "Element") -> str:
        return self._process_children(element)

    def _handle_org_name(self, element: "Element") -> str:
        return self._process_children(element)

    def _handle_date(self, element: "Element") -> str:
        return self._process_children(element)

    def _handle_name(self, element: "Element") -> str:
        return self._process_children(element)

    def _handle_table(self, element: "Element") -> str:
        if self.output_format == "clean":
            rows = []
            for row in element.findall(".//row"):
                cells = [self._process_element(cell).replace("\n", " ").strip() for cell in row.findall(".//cell")]
                if cells:
                    rows.append(" ".join(cells))
            return ". ".join(rows) + ". " if rows else ""
        return self._process_children(element)

    def _handle_table_row(self, element: "Element") -> str:
        return self._process_children(element)

    def _handle_table_cell(self, element: "Element") -> str:
        return self._process_children(element).replace("\n", " ").strip()

    def _handle_bibliography(self, element: "Element") -> str:
        return "" if self.output_format == "clean" else self._process_children(element)

    def _handle_bibl_struct(self, element: "Element") -> str:
        return self._process_children(element)

    def _handle_author(self, element: "Element") -> str:
        return self._process_children(element)

    def _handle_editor(self, element: "Element") -> str:
        return self._process_children(element)

    def _handle_pub_place(self, element: "Element") -> str:
        return self._process_children(element)

    def _handle_publisher(self, element: "Element") -> str:
        return self._process_children(element)

    def _handle_foreign(self, element: "Element") -> str:
        return self._process_children(element)

    def _handle_gap(self, element: "Element") -> str:
        return "" if self.output_format == "clean" else "[gap]"

    def _handle_unclear(self, element: "Element") -> str:
        content = self._process_children(element)
        return content if self.output_format == "clean" else f"[{content}?]"

    def _handle_supplied(self, element: "Element") -> str:
        return "" if self.output_format == "clean" else f"[{self._process_children(element)}]"

    def _handle_choice(self, element: "Element") -> str:
        from lxml import etree  # nosec

        corr = element.find("corr")
        expan = element.find("expan")

        if corr is not None:
            return self._process_element(corr)
        elif expan is not None:
            return self._process_element(expan)
        else:
            if self.output_format == "clean":
                for child in element:
                    tag = etree.QName(child).localname
                    if tag not in ["sic", "abbr"]:
                        return self._process_element(child)
                return ""
            return self._process_children(element)

    def _handle_apparatus(self, element: "Element") -> str:
        if self.output_format == "clean":
            lem = element.find("lem")
            if lem is not None:
                return self._process_element(lem)
            rdg = element.find("rdg")
            if rdg is not None:
                return self._process_element(rdg)
            return ""
        return self._process_children(element)

    def _handle_reading(self, element: "Element") -> str:
        return self._process_children(element)

    def _handle_sic(self, element: "Element") -> str:
        return self._process_children(element)

    def _handle_correction(self, element: "Element") -> str:
        return self._process_children(element)

    def _handle_abbreviation(self, element: "Element") -> str:
        return self._process_children(element)

    def _handle_expansion(self, element: "Element") -> str:
        return self._process_children(element)


@components.add("extract", "tei")
class TEIExtractor(MapFn):
    """Extract text from TEI XML format.

    This processor extracts plain text from TEI (Text Encoding Initiative) XML documents,
    with options for different output formats and handling of TEI-specific markup.
    """

    # Override base fields with specific _defaults
    name: str = Field(default="tei_extractor", description="Name of the TEI extractor")
    on: str = Field(default="text", description="Column containing TEI XML content")
    to: str = Field(default="text", description="Column to write extracted text to")

    # TEI-specific configuration
    output_format: Literal["markdown", "clean"] = Field(
        default="clean",
        description="Output format: 'clean' for plain text, 'markdown' for markdown with formatting",
    )

    # Private attributes (not part of the pydantic model validation)
    _parser: TEIParser = PrivateAttr()

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self._parser = TEIParser(output_format=self.output_format)

    def __call__(self, row: Row) -> Row:
        """Parse plain text from TEI-encoded XML."""
        xml_content = get_field(row, self.on) or ""
        text_content = self._parser(xml_content)
        set_field(row, self.to, text_content)
        return row
