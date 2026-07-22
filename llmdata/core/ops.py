from abc import ABC, abstractmethod
from typing import Any, TypeAlias

from ray.data.aggregate import AggregateFnV2

Row: TypeAlias = dict[str, Any]


class ReduceFn(AggregateFnV2):
    """Abstract base class for reduce operations."""


class MapFn(ABC):
    """Abstract base class for map operations."""

    def __init__(self, name: str, *on: str) -> None:
        self.name = name
        self.on = on

    @abstractmethod
    def __call__(self, row: Row) -> Row:
        """Read a row and returns the row with the map operation applied to it."""
        raise NotImplementedError


class FilterFn(ABC):
    """Abstract base class for filter operations."""

    def __init__(self, name: str, *on: str, if_missing: bool = True) -> None:
        self.name = name
        self.on = on
        self.if_missing = if_missing

    @abstractmethod
    def __call__(self, row: Row) -> bool:
        """Read a row and returns a boolean value for filtering."""
        raise NotImplementedError
