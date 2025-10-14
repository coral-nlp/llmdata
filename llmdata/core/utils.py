import logging
from contextlib import contextmanager
from typing import Any, Generator


def set_field(row: dict[str, Any], field: str, value: Any) -> None:
    """Set a value in a nested dictionary using dot notation.

    Creates intermediate dictionaries as needed. If any intermediate
    value exists and is not a dict, raises TypeError.

    Args:
        row: The dictionary to modify
        field: Dot-separated path (e.g., "metadata.language.score")
        value: The value to set

    Raises:
        TypeError: If an intermediate path component exists but is not a dict
        ValueError: If field is empty

    Example:
        >>> data = {}
        >>> set_field(data, "metadata.language.score", 0.95)
        >>> print(data)
        {'metadata': {'language': {'score': 0.95}}}
    """
    if not field:
        raise ValueError("Field cannot be empty")

    keys = field.split(".")
    current = row

    # Navigate to the parent of the target key
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        elif not isinstance(current[key], dict):
            raise TypeError(
                f"Cannot set field '{field}': intermediate key '{key}' "
                f"exists but is not a dict (type: {type(current[key]).__name__})"
            )
        current = current[key]

    # Set the final value
    current[keys[-1]] = value


def get_field(row: dict[str, Any], field: str) -> Any | None:
    """Get a value from a nested dictionary using dot notation.

    Returns None if any part of the path doesn't exist or if an
    intermediate value is not a dictionary.

    Args:
        row: The dictionary to read from
        field: Dot-separated path (e.g., "metadata.language.score")

    Returns:
        The value at the specified path, or None if not found

    Example:
        >>> data = {"metadata": {"language": {"score": 0.95}}}
        >>> get_field(data, "metadata.language.score")
        0.95
        >>> get_field(data, "metadata.missing.key")
        None
    """
    if not field:
        return None

    keys = field.split(".")
    current = row

    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]

    return current


@contextmanager
def silence(highest_level: Any = logging.CRITICAL) -> Generator[None, Any, None]:
    """Silence all warnings and logging in the current context."""
    previous_level = logging.root.manager.disable
    logging.disable(highest_level)
    try:
        yield
    finally:
        logging.disable(previous_level)
