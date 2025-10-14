import json
from typing import Any, Callable

from pydantic import BaseModel

from .dependencies import Dependency


class Registry:
    """Registry for LLMData components."""

    def __init__(self) -> None:
        # Format: {category: {type: class}}
        self._classes: dict[str, dict[str, type]] = {}
        self._schemas: dict[str, dict[str, dict[str, Any]]] = {}

    def add(self, category: str, identifier: str) -> Callable[[type], type[BaseModel] | type]:
        """Register a component class for a specific format."""

        def decorator(cls: type) -> type:
            if category not in self._classes:
                self._classes[category] = {identifier: cls}
            else:
                self._classes[category][identifier] = cls

            # Generate and cache schema if it's a Pydantic model
            if issubclass(cls, BaseModel):
                self._generate_schema(category, identifier, cls)

            return cls

        return decorator

    def _generate_schema(self, category: str, identifier: str, cls: type[BaseModel]) -> None:
        """Generate JSON schema for a Pydantic model."""
        try:
            schema = cls.model_json_schema()
            schema["$id"] = f"llmdata://{category}/{identifier}"
            schema["title"] = f"{category.capitalize()} {identifier.capitalize()}"

            if category not in self._schemas:
                self._schemas[category] = {}
            self._schemas[category][identifier] = schema
        except Exception as e:
            print(f"Warning: Could not generate schema for {category}.{identifier}: {e}")

    def get(self, category: str, identifier: str) -> type:
        """Get component class for specific category and type."""
        try:
            cats = self._classes[category]
        except KeyError:
            raise ValueError(f"Unknown category '{category}'. Available formats: {list(self._classes.keys())}")
        try:
            return cats[identifier]
        except KeyError:
            raise ValueError(f"Unknown type '{identifier}' in category '{category}'")

    def get_schema(self, category: str, identifier: str) -> dict[str, Any]:
        """Get JSON schema for a specific component."""
        try:
            return self._schemas[category][identifier]
        except KeyError:
            raise ValueError(f"No schema available for {category}.{identifier}")

    def get_all_schemas(self, category: str | None = None) -> dict[str, Any]:
        """Get all schemas, optionally filtered by category."""
        if category:
            return self._schemas.get(category, {})
        return self._schemas

    def export_schemas(self, output_path: str, category: str | None = None) -> None:
        """Export schemas to JSON file."""
        schemas = self.get_all_schemas(category)
        with open(output_path, "w") as f:
            json.dump(schemas, f, indent=2)

    def categories(self) -> list[str]:
        """Return registered category names."""
        return list(self._classes.keys())

    def components(self, category: str) -> list[str]:
        """Return registered component names for a category."""
        try:
            return list(self._classes[category].keys())
        except KeyError:
            raise ValueError(f"Unknown category '{category}'")

    def has(self, category: str, identifier: str) -> bool:
        """Check if a given type is supported in a given category."""
        return category in self._classes and identifier in self._classes[category]

    def validate_config(self, category: str, identifier: str, config: dict[str, Any]) -> dict[str, Any]:
        """Validate configuration against component schema."""
        cls = self.get(category, identifier)
        if issubclass(cls, BaseModel):
            # Use Pydantic validation
            instance = cls.model_validate(config)  # type: ignore[attr-defined]
            config = instance.model_dump()
        return config

    def get_dependencies(self, category: str, identifier: str) -> list[Dependency]:
        """Get dependencies for a specific component."""
        cls = self.get(category, identifier)
        if hasattr(cls, "_required_dependencies"):
            return cls._required_dependencies  # type: ignore[no-any-return]
        return []

    def get_all_dependencies_for_categories(self, category_types: list[tuple[str, str]]) -> list[Dependency]:
        """Get all unique dependencies for a list of (category, type) pairs."""
        all_deps = []
        seen_modules = set()

        for category, identifier in category_types:
            deps = self.get_dependencies(category, identifier)
            for dep in deps:
                # Use module name as unique identifier to avoid duplicates
                if dep.module not in seen_modules:
                    all_deps.append(dep)
                    seen_modules.add(dep.module)

        return all_deps


components = Registry()
