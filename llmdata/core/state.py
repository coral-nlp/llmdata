from abc import ABC, abstractmethod
from typing import Any

import ray
from pyarrow._fs import FileType

from .filesystem import get_fs


class BaseState(ABC):
    """Abstraction for ray actor state."""

    @abstractmethod
    def exists(self) -> bool:
        """Check if state exists."""
        raise NotImplementedError

    @abstractmethod
    def save(self, data: Any) -> None:
        """Save state."""
        raise NotImplementedError

    @abstractmethod
    def restore(self) -> Any | None:
        """Restore state."""
        raise NotImplementedError


@ray.remote
class FileState(BaseState):
    """File-based persistence provider."""

    def __init__(self, file: str = "state.pkl") -> None:
        self.file = file
        self.fs = get_fs(file, "fsspec")

    def exists(self) -> bool:
        """Check if state file exists."""
        info = self.fs.get_file_info(self.file)
        if isinstance(info, list):
            return True
        return bool(info.type != FileType.NotFound)

    def save(self, data: Any) -> None:
        """Save state to file."""
        state = ray.cloudpickle.dumps(data)
        with self.fs.open_(self.file, "wb") as f:
            f.write(state)

    def restore(self) -> Any | None:
        """Restore state from file."""
        if not self.exists():
            return None
        else:
            with self.fs.open(self.file, "rb") as f:
                state = f.read()
            return ray.cloudpickle.loads(state)
