from typing import Any, Literal

from pydantic import Field

from llmdata.core.ops import FilterFn, Row
from llmdata.core.registry import components
from llmdata.core.utils import get_field

from .language import LanguageFilter
from .quality import GopherQualityFilter, GopherRepetitionFilter
from .tokens import TokenCountFilter

# @components.add("filter", "lambda")
# class LambdaFilter(FilterFn):
#    """Generic filter that accepts a custom predicate function."""
#
#    fn: Callable[[Any], bool] = Field(description="Custom predicate function to apply")
#
#    def __call__(self, row: Row) -> bool:
#        """Applies the lambda predicate to the row and returns the result."""
#        val: Any = get_field(row, self.on)
#        if not val:
#            return self.if_missing
#        return self.fn(val)


@components.add("filter", "value")
class ValueFilter(FilterFn):
    """Filter for comparing against a given value. Supports multiple comparators."""

    value: Any = Field(description="Value to compare against")
    comparator: Literal["eq", "neq", "gt", "lt", "gte", "lte", "inl", "inr", "ninl", "ninr"] = Field(
        default="eq", description="Comparison operator"
    )

    def __call__(self, row: Row) -> bool:
        """Compare a given value to the desired value using the specified comparator."""
        got = get_field(row, self.on)
        if not got:
            return self.if_missing
        if self.comparator == "eq":
            return got == self.value or self.if_missing
        elif self.comparator == "neq":
            return got != self.value or self.if_missing
        elif self.comparator == "gt":
            return got > self.value or self.if_missing
        elif self.comparator == "lt":
            return got < self.value or self.if_missing
        elif self.comparator == "gte":
            return got >= self.value or self.if_missing
        elif self.comparator == "lte":
            return got <= self.value or self.if_missing
        elif self.comparator == "inr":
            return got in self.value or self.if_missing
        elif self.comparator == "inl":
            return self.value in got or self.if_missing
        elif self.comparator == "ninr":
            return got not in self.value or self.if_missing
        elif self.comparator == "ninl":
            return self.value not in got or self.if_missing
        else:
            return self.if_missing


@components.add("filter", "exists")
class ExistsFilter(FilterFn):
    """Filter based on presence of fields."""

    def __call__(self, row: Row) -> bool:
        """Check if the specified field exists in the given row."""
        return get_field(row, self.on) is not None


__all__ = [
    "ExistsFilter",
    "GopherQualityFilter",
    "GopherRepetitionFilter",
    # "LambdaFilter",
    "LanguageFilter",
    "TokenCountFilter",
    "ValueFilter",
]
