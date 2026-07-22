from typing import TYPE_CHECKING, Any

import ray

from .registry import components

if TYPE_CHECKING:
    from ray.data import Dataset as RayDataset

    from .config import RayConfig


class Reader:
    """Base Ray data reader."""

    def __init__(self, config: "RayConfig", filesystem: Any = None, **kwargs: Any) -> None:
        self.config = config
        self.filesystem = filesystem
        self.params = kwargs

    def __call__(self, path: str | list[str]) -> "RayDataset":
        """Read data and return Ray dataset."""
        raise NotImplementedError


@components.add("reader", "parquet")
class ParquetReader(Reader):
    """Reader for parquet files."""

    def __call__(self, path: str | list[str]) -> "RayDataset":
        """Read parquet data and return ray dataset."""
        read_kwargs = self.config.get_read_kwargs()
        read_kwargs.update(self.params)
        return ray.data.read_parquet(path, filesystem=self.filesystem, **read_kwargs)


@components.add("reader", "jsonl")
class JSONLReader(Reader):
    """Reader for JSONL files."""

    def __call__(self, path: str | list[str]) -> "RayDataset":
        """Read jsonl data and return ray dataset."""
        read_kwargs = self.config.get_read_kwargs()
        read_kwargs.update(self.params)
        return ray.data.read_json(path, filesystem=self.filesystem, **read_kwargs)


@components.add("reader", "csv")
class CSVReader(Reader):
    """Reader for CSV files."""

    def __call__(self, path: str | list[str]) -> "RayDataset":
        """Read CSV data and return ray dataset."""
        read_kwargs = self.config.get_read_kwargs()
        read_kwargs.update(self.params)
        return ray.data.read_csv(path, filesystem=self.filesystem, **read_kwargs)


@components.add("reader", "text")
class TextReader(Reader):
    """Reader for plain text files."""

    def __call__(self, path: str | list[str]) -> "RayDataset":
        """Read text files line by line and return ray dataset."""
        read_kwargs = self.config.get_read_kwargs()
        read_kwargs.update(self.params)
        return ray.data.read_text(path, filesystem=self.filesystem, **read_kwargs)
