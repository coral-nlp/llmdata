import os
from typing import TYPE_CHECKING, Callable, Literal
from urllib.parse import urlparse

if TYPE_CHECKING:
    # Pyarrow fs typehints; renamed to avoid collisions
    # Fsspec s3 typehints; renamed to avoid collisions
    from fsspec import AbstractFileSystem as FsFileSystem
    from fsspec.implementations.local import FsLocalFileSystem
    from pyarrow.fs import FileSystem as ArrowFileSystem
    from pyarrow.fs import LocalFileSystem as ArrowLocalFileSystem
    from pyarrow.fs import S3FileSystem as ArrowS3FileSystem
    from s3fs import S3FileSystem as FsS3FileSystem


def get_s3_fs_pyarrow() -> "ArrowS3FileSystem":
    """Create a S3 filesystem from environment variables using pyarrow."""
    from pyarrow.fs import S3FileSystem

    return S3FileSystem(
        access_key=os.environ["AWS_ACCESS_KEY_ID"],
        secret_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        endpoint_override=os.environ["AWS_ENDPOINT_URL"],
        allow_bucket_creation=False,
        allow_bucket_deletion=False,
        scheme=os.environ["AWS_SECURE_SCHEME"],
    )


def get_local_fs_pyarrow() -> "ArrowLocalFileSystem":
    """Create a local filesystem using pyarrow."""
    from pyarrow.fs import LocalFileSystem

    return LocalFileSystem()


# Filesystem mapping for pyarrow-backed filesystems
FS_PYARROW: "dict[str, Callable[[],ArrowFileSystem]]" = {
    "file": get_local_fs_pyarrow,
    "s3": get_s3_fs_pyarrow,
}


def get_local_fs_fsspec() -> "FsLocalFileSystem":
    """Create a local filesystem using fsspec."""
    from fsspec.implementations.local import LocalFileSystem

    return LocalFileSystem()


def get_s3_fs_fsspec() -> "FsS3FileSystem":
    """Create a S3 filesystem from environment variables using fsspec."""
    from s3fs import S3FileSystem

    return S3FileSystem(
        key=os.environ["AWS_ACCESS_KEY_ID"],
        secret=os.environ["AWS_SECRET_ACCESS_KEY"],
        endpoint_url=os.environ["AWS_ENDPOINT_URL"],
        use_ssl=os.environ["AWS_SECURE_SCHEME"] == "https",
    )


# Filesystem mapping for fsspec-backed filesystems
FS_FSSPEC: "dict[str, Callable[[], FsFileSystem]]" = {
    "file": get_local_fs_fsspec,
    "s3": get_s3_fs_fsspec,
}


def get_fs(
    path: str | list[str], backend: Literal["pyarrow", "fsspec"] = "pyarrow"
) -> "ArrowFileSystem | FsFileSystem":
    """Get configured filesystem for path."""
    scheme = urlparse(path).scheme or "file" if isinstance(path, str) else urlparse(path[0]).scheme or "file"
    if backend == "pyarrow":
        if scheme not in FS_PYARROW:
            raise ValueError(f"Unsupported scheme {scheme} for backend {backend}")
        return FS_PYARROW[scheme]()
    elif backend == "fsspec":
        if scheme not in FS_FSSPEC:
            raise ValueError(f"Unsupported scheme {scheme} for backend {backend}")
        return FS_FSSPEC[scheme]()
    else:
        raise ValueError(f"Unsupported backend {backend}")
