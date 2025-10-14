from typing import TYPE_CHECKING, Any

from .config import get_default_ray_config
from .filesystem import get_fs
from .registry import components

if TYPE_CHECKING:
    from ray.data import Dataset

    from .config import RayConfig


class Writer:
    """Base Ray data writer."""

    def __init__(
        self,
        config: "RayConfig | None" = None,
        **kwargs: dict[str, Any],
    ) -> None:
        self.config = config or get_default_ray_config()
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

        # Handle parquet-specific parameters
        for key in ["compression", "row_group_size", "partition_cols"]:
            if key in self.params:
                if key == "partition_cols" and isinstance(self.params[key], str):
                    write_kwargs[key] = [self.params[key]]
                else:
                    write_kwargs[key] = self.params[key]

        # Set default compression if not specified
        if "compression" not in write_kwargs:
            write_kwargs["compression"] = "snappy"

        dataset.write_parquet(path, filesystem=get_fs(path, "pyarrow"), **write_kwargs)


@components.add("writer", "jsonl")
class JSONLWriter(Writer):
    """Writer for JSONL files."""

    def __call__(self, dataset: "Dataset", path: str) -> None:
        """Write dataset to jsonl file."""
        write_kwargs = self.config.get_write_kwargs()

        # Handle JSONL-specific parameters
        for key in ["lines_delimiter", "force_ascii"]:
            if key in self.params:
                write_kwargs[key] = self.params[key]

        dataset.write_json(path, filesystem=get_fs(path, "pyarrow"), **write_kwargs)


@components.add("writer", "csv")
class CSVWriter(Writer):
    """Writer for CSV files."""

    def __call__(self, dataset: "Dataset", path: str) -> None:
        """Write dataset to csv file."""
        write_kwargs = self.config.get_write_kwargs()

        # Handle CSV-specific parameters
        for key in ["delimiter", "header", "include_header", "escape_char", "quote_char"]:
            if key in self.params:
                write_kwargs[key] = self.params[key]

        # Set _defaults
        if "include_header" not in write_kwargs:
            write_kwargs["include_header"] = True

        dataset.write_csv(path, filesystem=get_fs(path, "pyarrow"), **write_kwargs)
