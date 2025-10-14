import importlib
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, TypeVar

# Type variable for decorated classes
T = TypeVar("T", bound=type)


@dataclass
class Dependency:
    """Represents a single package dependency requirement."""

    module: str
    package: str | None = None
    min_version: str | None = None
    install_hint: str | None = None
    check_attr: str | None = None

    def __post_init__(self) -> None:  # noqa
        # Auto-derive package name from module if not provided
        if self.package is None:
            self.package = self.module.split(".")[0]

        # Auto-generate install hint if not provided
        if self.install_hint is None:
            if self.min_version:
                self.install_hint = f"pip install {self.package}>={self.min_version}"
            else:
                self.install_hint = f"pip install {self.package}"

    def is_available(self) -> bool:
        """Check if the dependency is available and meets requirements."""
        try:
            module = importlib.import_module(self.module)
            return not (self.check_attr and not hasattr(module, self.check_attr))
        except ImportError:
            return False

    def get_error_message(self) -> str:
        """Get a descriptive error message for missing dependency."""
        base_msg = f"Missing required dependency: {self.package}"
        if self.min_version:
            base_msg += f" (>= {self.min_version})"
        return f"{base_msg}. Install with: {self.install_hint}"


def requires(*dependencies: str | Dependency) -> Callable[[T], T]:
    """Class decorator that checks for required dependencies at runtime.

    This decorator replaces the need for manual try/except ImportError blocks
    by automatically checking dependencies when the class is instantiated.

    Args:
        *dependencies: Package requirements. Can be:
            - str: Simple module name (e.g., "fasttext")
            - Dependency: Full dependency specification with version, install hints, etc.

    Returns:
        Decorated class with dependency checking

    Examples:
        >>> @requires("fasttext")
        ... class LanguageTagger:
        ...     pass

        >>> @requires(
        ...     Dependency("transformers", min_version="4.0.0"),
        ...     Dependency("torch", package="torch", install_hint="pip install torch --index-url https://download.pytorch.org/whl/cpu")
        ... )
        ... class TokenCountTagger:
        ...     pass

        >>> # For packages with complex import paths
        ... @requires(
        ...     Dependency("presidio_analyzer", package="presidio-analyzer", install_hint="pip install presidio-analyzer presidio-anonymizer")
        ... )
        ... class PresidioPIIFormatter:
        ...     pass

    """

    def decorator(cls: T) -> T:
        # Convert string dependencies to Dependency objects
        dep_objects = []
        for dep in dependencies:
            if isinstance(dep, str):
                dep_objects.append(Dependency(dep))
            else:
                dep_objects.append(dep)

        # Store original __init__ method
        original_init = cls.__init__  # type: ignore[misc]

        @wraps(original_init)
        def checked_init(self: Any, *args: Any, **kwargs: Any) -> None:
            # Check all dependencies before initialization
            missing_deps = []
            for dep in dep_objects:
                if not dep.is_available():
                    missing_deps.append(dep)

            if missing_deps:
                # Create comprehensive error message
                if len(missing_deps) == 1:
                    raise ImportError(missing_deps[0].get_error_message())
                else:
                    error_lines = [f"Missing required dependencies for {cls.__name__}:"]
                    for dep in missing_deps:
                        error_lines.append(f"  - {dep.get_error_message()}")
                    raise ImportError("\n".join(error_lines))

            # Call original __init__ if all dependencies are available
            original_init(self, *args, **kwargs)

        # Replace __init__ with the checked version
        cls.__init__ = checked_init  # type: ignore[misc]

        # Store dependency information on the class for introspection
        cls._required_dependencies = dep_objects  # type: ignore[attr-defined]

        return cls

    return decorator


def check_dependencies(*dependencies: str | Dependency) -> bool:
    """Check if all specified dependencies are available without raising an error.

    Args:
        *dependencies: Package requirements to check

    Returns:
        True if all dependencies are available, False otherwise

    Example:
        >>> if check_dependencies("fasttext", "transformers"):
        ...     print("All dependencies available!")
        ... else:
        ...     print("Some dependencies missing")

    """
    dep_objects = []
    for dep in dependencies:
        if isinstance(dep, str):
            dep_objects.append(Dependency(dep))
        else:
            dep_objects.append(dep)

    return all(dep.is_available() for dep in dep_objects)


def get_missing_dependencies(*dependencies: str | Dependency) -> list[Dependency]:
    """Get a list of missing dependencies from the specified requirements.

    Args:
        *dependencies: Package requirements to check

    Returns:
        List of missing Dependency objects

    Example:
        >>> missing = get_missing_dependencies("fasttext", "nonexistent_package")
        >>> for dep in missing:
        ...     print(f"Missing: {dep.package}")

    """
    dep_objects = []
    for dep in dependencies:
        if isinstance(dep, str):
            dep_objects.append(Dependency(dep))
        else:
            dep_objects.append(dep)

    return [dep for dep in dep_objects if not dep.is_available()]
