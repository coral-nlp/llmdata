"""Tests for the LLMData components system."""

import pytest

from llmdata.core.registry import Registry


class TestRegistry:
    """Test the Registry class functionality."""

    def setup_method(self):
        """Set up a fresh components for each test."""
        self.registry = Registry()

    def test_registry_initialization(self):
        """Test components initializes empty."""
        assert self.registry.categories() == []

    def test_add_component_decorator(self):
        """Test adding components via decorator."""

        @self.registry.add("reader", "test_format")
        class TestReader:
            pass

        assert "reader" in self.registry.categories()
        assert "test_format" in self.registry.components("reader")
        assert self.registry.get("reader", "test_format") == TestReader

    def test_add_multiple_components(self):
        """Test adding multiple components to same category."""

        @self.registry.add("reader", "format1")
        class Reader1:
            pass

        @self.registry.add("reader", "format2")
        class Reader2:
            pass

        components = self.registry.components("reader")
        assert "format1" in components
        assert "format2" in components
        assert len(components) == 2

    def test_add_components_different_categories(self):
        """Test adding components to different categories."""

        @self.registry.add("reader", "parquet")
        class ParquetReader:
            pass

        @self.registry.add("writer", "parquet")
        class ParquetWriter:
            pass

        categories = self.registry.categories()
        assert "reader" in categories
        assert "writer" in categories
        assert len(categories) == 2

    def test_get_component(self):
        """Test retrieving registered components."""

        @self.registry.add("filter", "language")
        class LanguageFilter:
            pass

        retrieved = self.registry.get("filter", "language")
        assert retrieved == LanguageFilter

    def test_get_nonexistent_category(self):
        """Test error when getting from nonexistent category."""
        with pytest.raises(ValueError, match="Unknown category 'nonexistent'"):
            self.registry.get("nonexistent", "format")

    def test_get_nonexistent_component(self):
        """Test error when getting nonexistent component."""

        @self.registry.add("reader", "parquet")
        class ParquetReader:
            pass

        with pytest.raises(ValueError, match="Unknown type 'nonexistent' in category 'reader'"):
            self.registry.get("reader", "nonexistent")

    def test_has_component(self):
        """Test checking if component exists."""

        @self.registry.add("writer", "csv")
        class CSVWriter:
            pass

        assert self.registry.has("writer", "csv") is True
        assert self.registry.has("writer", "parquet") is False
        assert self.registry.has("reader", "csv") is False

    def test_components_for_category(self):
        """Test listing components for specific category."""

        @self.registry.add("tagger", "language")
        class LanguageTagger:
            pass

        @self.registry.add("tagger", "sentiment")
        class SentimentTagger:
            pass

        components = self.registry.components("tagger")
        assert "language" in components
        assert "sentiment" in components
        assert len(components) == 2

    def test_components_nonexistent_category(self):
        """Test error when listing components for nonexistent category."""
        with pytest.raises(ValueError, match="Unknown category 'nonexistent'"):
            self.registry.components("nonexistent")

    def test_decorator_returns_class(self):
        """Test that decorator returns the original class."""
        original_class = type("TestClass", (), {})
        decorated_class = self.registry.add("test", "format")(original_class)
        assert decorated_class is original_class

    def test_overwrite_existing_component(self):
        """Test overwriting existing component registration."""

        @self.registry.add("reader", "json")
        class JSONReader1:
            pass

        @self.registry.add("reader", "json")
        class JSONReader2:
            pass

        # Should return the latest registered class
        assert self.registry.get("reader", "json") == JSONReader2


class TestGlobalRegistry:
    """Test the global components instance."""

    def test_global_registry_exists(self):
        """Test that global components instance exists."""
        from llmdata.core.registry import components

        assert isinstance(components, Registry)

    def test_global_registry_shared_state(self):
        """Test that global components maintains state across imports."""
        from llmdata.core.registry import components as registry1
        from llmdata.core.registry import components as registry2

        @registry1.add("test", "shared")
        class SharedClass:
            pass

        # Should be accessible from second import
        assert registry2.has("test", "shared")
        assert registry2.get("test", "shared") == SharedClass


class TestRegistryIntegration:
    """Test components integration with actual components."""

    def test_reader_registration(self):
        """Test that readers are properly registered."""
        from llmdata.core.registry import components

        # Check that readers are registered
        assert components.has("reader", "parquet")
        assert components.has("reader", "jsonl")
        assert components.has("reader", "csv")
        assert components.has("reader", "text")

    def test_writer_registration(self):
        """Test that writers are properly registered."""
        from llmdata.core.registry import components

        # Check that writers are registered
        assert components.has("writer", "parquet")
        assert components.has("writer", "jsonl")
        assert components.has("writer", "csv")

    def test_component_instantiation(self):
        """Test that registered components can be instantiated."""
        from llmdata.core.config import get_default_ray_config
        from llmdata.core.registry import components

        # Get a reader class and instantiate it
        reader_cls = components.get("reader", "parquet")
        config = get_default_ray_config()
        reader = reader_cls(config)

        assert reader is not None
        assert callable(reader)

    def test_components_list_formats(self):
        """Test listing available formats for components."""
        from llmdata.core.registry import components

        reader_formats = components.components("reader")
        writer_formats = components.components("writer")

        # Should have multiple formats
        assert len(reader_formats) >= 3
        assert len(writer_formats) >= 3

        # Common formats should exist
        assert "parquet" in reader_formats
        assert "parquet" in writer_formats
