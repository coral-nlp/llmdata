from typing import TYPE_CHECKING, Any

from .config import get_default_ray_config
from .registry import components

if TYPE_CHECKING:
    from ray.data import Dataset

    from .config import RayConfig


class Writer:
    """Base Ray data writer."""

    def __init__(
        self,
        config: "RayConfig | None" = None,
        filesystem: Any = None,
        **kwargs: Any,
    ) -> None:
        self.config = config or get_default_ray_config()
        self.filesystem = filesystem
        self.params = kwargs

    def __call__(self, dataset: "Dataset", path: str) -> None:
        """Write dataset to file."""
        raise NotImplementedError


@components.add("writer", "parquet")
class ParquetWriter(Writer):
    """Writer for Parquet files."""

    def __call__(self, dataset: "Dataset", path: str) -> None:
        """Write dataset to parquet file."""
        write_kwargs = self.config.get_write_kwargs()
        write_kwargs.update(self.params)
        if "compression" not in write_kwargs:
            write_kwargs["compression"] = "snappy"
        dataset.write_parquet(path, filesystem=self.filesystem, **write_kwargs)


@components.add("writer", "jsonl")
class JSONLWriter(Writer):
    """Writer for JSONL files."""

    def __call__(self, dataset: "Dataset", path: str) -> None:
        """Write dataset to jsonl file."""
        write_kwargs = self.config.get_write_kwargs()
        write_kwargs.update(self.params)
        dataset.write_json(path, filesystem=self.filesystem, **write_kwargs)


@components.add("writer", "csv")
class CSVWriter(Writer):
    """Writer for CSV files."""

    def __call__(self, dataset: "Dataset", path: str) -> None:
        """Write dataset to csv file."""
        write_kwargs = self.config.get_write_kwargs()
        write_kwargs.update(self.params)
        if "include_header" not in write_kwargs:
            write_kwargs["include_header"] = True
        dataset.write_csv(path, filesystem=self.filesystem, **write_kwargs)
