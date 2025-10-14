from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, TypeAlias

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from ray.data.aggregate import AggregateFnV2


Row: TypeAlias = dict[str, Any]


class ReduceFn(ABC, BaseModel):
    """Abstract base class for reduce operations."""

    name: str = Field(title="The name of this reduce function.")
    on: str = Field(title="The column to apply this function to.")

    @abstractmethod
    def __call__(self) -> "AggregateFnV2":
        """Return a compatible aggregation function for ray to consume."""
        raise NotImplementedError


class MapFn(ABC, BaseModel):
    """Abstract base class for map operations."""

    name: str = Field(title="The name of this map function.")
    on: str = Field(title="The column to read input data from for the map operation.")
    to: str = Field(title="The column to write the results of the map operation to.")

    @abstractmethod
    def __call__(self, row: Row) -> Row:
        """Read a row and returns the row with the map operation applied to it."""
        raise NotImplementedError


class FilterFn(ABC, BaseModel):
    """Abstract base class for filter operations."""

    name: str = Field(title="The name of this filter function.")
    on: str = Field(title="The column to apply this filter function to.")
    if_missing: bool = Field(default=True, title="The value the filter returns if encountering a missing value.")

    @abstractmethod
    def __call__(self, row: Row) -> bool:
        """Read a row and returns a boolean value for filtering."""
        raise NotImplementedError
