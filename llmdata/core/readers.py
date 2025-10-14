import glob
from typing import TYPE_CHECKING, Any

import ray
from fsspec.implementations.local import LocalFileSystem as FsLocalFileSystem
from pyarrow.fs import LocalFileSystem as ArrowLocalFileSystem

from .filesystem import get_fs
from .registry import components

if TYPE_CHECKING:
    from ray.data import Dataset as RayDataset

    from .config import RayConfig


class Reader:
    """Base Ray data reader."""

    def __init__(self, config: "RayConfig", **kwargs: dict[str, Any]) -> None:
        self.config = config
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

        # Handle parquet-specific parameters
        if "columns" in self.params:
            read_kwargs["columns"] = self.params["columns"]
        if "batch_size" in self.params:
            read_kwargs["batch_size"] = self.params["batch_size"]

        fs = get_fs(path, "pyarrow")
        if "*" in path and isinstance(path, str):
            if not isinstance(fs, FsLocalFileSystem | ArrowLocalFileSystem):
                raise ValueError("Wildcard paths only supported for local filesystems")
            path = glob.glob(path)

        ds: RayDataset = ray.data.read_parquet(path, filesystem=fs, **read_kwargs)
        ds = ds.select_columns(["id", "subset", "source", "text", "license", "num_tokens"])
        return ds


@components.add("reader", "jsonl")
class JSONLReader(Reader):
    """Reader for JSONL files."""

    def __call__(self, path: str | list[str]) -> "RayDataset":
        """Read jsonl data and return ray dataset."""
        read_kwargs = self.config.get_read_kwargs()
        fs = get_fs(path, "pyarrow")
        if "*" in path and isinstance(path, str):
            if not isinstance(fs, FsLocalFileSystem | ArrowLocalFileSystem):
                raise ValueError("Wildcard paths only supported for local filesystems")
            path = glob.glob(path)

        ds: RayDataset = ray.data.read_json(path, filesystem=get_fs(path, "pyarrow"), **read_kwargs)
        return ds


@components.add("reader", "csv")
class CSVReader(Reader):
    """Reader for CSV files."""

    def __call__(self, path: str | list[str]) -> "RayDataset":
        """Read CSV data and return ray dataset."""
        read_kwargs = self.config.get_read_kwargs()

        # Handle CSV-specific parameters
        for key in ["delimiter", "header", "names", "dtype", "usecols", "skiprows"]:
            if key in self.params:
                read_kwargs[key] = self.params[key]

        fs = get_fs(path, "pyarrow")
        if "*" in path and isinstance(path, str):
            if not isinstance(fs, FsLocalFileSystem | ArrowLocalFileSystem):
                raise ValueError("Wildcard paths only supported for local filesystems")
            path = glob.glob(path)

        return ray.data.read_csv(path, filesystem=get_fs(path, "pyarrow"), **read_kwargs)


@components.add("reader", "text")
class TextReader(Reader):
    """Reader for plain text files."""

    def __call__(self, path: str | list[str]) -> "RayDataset":
        """Read text files line by line and return ray dataset."""
        read_kwargs = self.config.get_read_kwargs()

        # Handle text-specific parameters
        if "encoding" in self.params:
            read_kwargs["encoding"] = self.params["encoding"]

        fs = get_fs(path, "pyarrow")
        if "*" in path and isinstance(path, str):
            if not isinstance(fs, FsLocalFileSystem | ArrowLocalFileSystem):
                raise ValueError("Wildcard paths only supported for local filesystems")
            path = glob.glob(path)

        ds: RayDataset = ray.data.read_text(path, filesystem=get_fs(path, "pyarrow"), **read_kwargs)
        return ds
