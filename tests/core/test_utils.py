from typing import Any, Dict

import pytest

from llmdata.core.utils import get_field, set_field


class TestAddField:
    """Test cases for add_field function."""

    def test_add_simple_field(self):
        """Test adding a simple top-level field."""
        row = {}
        set_field(row, "key", "value")
        assert row == {"key": "value"}

    def test_add_nested_field_empty_dict(self):
        """Test adding nested field to empty dictionary."""
        row = {}
        set_field(row, "metadata.language.score", 0.95)
        expected = {"metadata": {"language": {"score": 0.95}}}
        assert row == expected

    def test_add_nested_field_existing_structure(self):
        """Test adding field to existing nested structure."""
        row = {"metadata": {"language": {"confidence": 0.87}}}
        set_field(row, "metadata.language.score", 0.95)
        expected = {"metadata": {"language": {"confidence": 0.87, "score": 0.95}}}
        assert row == expected

    def test_add_field_creates_parallel_branches(self):
        """Test adding fields that create parallel nested branches."""
        row = {}
        set_field(row, "metadata.language.score", 0.95)
        set_field(row, "metadata.processing.timestamp", "2025-06-19")
        set_field(row, "other.field", "value")

        expected = {
            "metadata": {"language": {"score": 0.95}, "processing": {"timestamp": "2025-06-19"}},
            "other": {"field": "value"},
        }
        assert row == expected

    def test_add_field_overwrites_existing_value(self):
        """Test that adding a field overwrites existing value."""
        row = {"metadata": {"score": 0.5}}
        set_field(row, "metadata.score", 0.95)
        assert row == {"metadata": {"score": 0.95}}

    def test_add_field_various_value_types(self):
        """Test adding fields with various value types."""
        row = {}
        set_field(row, "string", "test")
        set_field(row, "number", 42)
        set_field(row, "float", 3.14)
        set_field(row, "boolean", True)
        set_field(row, "none", None)
        set_field(row, "list", [1, 2, 3])
        set_field(row, "dict", {"nested": "value"})

        assert row["string"] == "test"
        assert row["number"] == 42
        assert row["float"] == 3.14
        assert row["boolean"] is True
        assert row["none"] is None
        assert row["list"] == [1, 2, 3]
        assert row["dict"] == {"nested": "value"}

    def test_add_field_deep_nesting(self):
        """Test adding field with deep nesting."""
        row = {}
        set_field(row, "a.b.c.d.e.f", "deep_value")

        current = row
        for key in ["a", "b", "c", "d", "e"]:
            assert isinstance(current[key], dict)
            current = current[key]
        assert current["f"] == "deep_value"

    def test_add_field_empty_string_raises_error(self):
        """Test that empty field string raises ValueError."""
        row = {}
        with pytest.raises(ValueError, match="Field cannot be empty"):
            set_field(row, "", "value")

    def test_add_field_intermediate_non_dict_raises_error(self):
        """Test that non-dict intermediate values raise TypeError."""
        row = {"metadata": "not_a_dict"}
        with pytest.raises(TypeError, match="intermediate key 'metadata' exists but is not a dict"):
            set_field(row, "metadata.language.score", 0.95)

    def test_add_field_intermediate_number_raises_error(self):
        """Test that numeric intermediate values raise TypeError."""
        row = {"metadata": {"score": 42}}
        with pytest.raises(TypeError, match="intermediate key 'score' exists but is not a dict"):
            set_field(row, "metadata.score.subscore", 0.95)

    def test_add_field_intermediate_list_raises_error(self):
        """Test that list intermediate values raise TypeError."""
        row = {"metadata": {"items": [1, 2, 3]}}
        with pytest.raises(TypeError, match="intermediate key 'items' exists but is not a dict"):
            set_field(row, "metadata.items.length", 3)


class TestGetField:
    """Test cases for get_field function."""

    def test_get_simple_field(self):
        """Test getting a simple top-level field."""
        row = {"key": "value"}
        assert get_field(row, "key") == "value"

    def test_get_nested_field(self):
        """Test getting nested field."""
        row = {"metadata": {"language": {"score": 0.95}}}
        assert get_field(row, "metadata.language.score") == 0.95

    def test_get_field_various_types(self):
        """Test getting fields with various value types."""
        row = {
            "string": "test",
            "number": 42,
            "float": 3.14,
            "boolean": True,
            "none": None,
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
        }

        assert get_field(row, "string") == "test"
        assert get_field(row, "number") == 42
        assert get_field(row, "float") == 3.14
        assert get_field(row, "boolean") is True
        assert get_field(row, "none") is None
        assert get_field(row, "list") == [1, 2, 3]
        assert get_field(row, "dict") == {"nested": "value"}

    def test_get_deep_nested_field(self):
        """Test getting deeply nested field."""
        row = {"a": {"b": {"c": {"d": {"e": {"f": "deep_value"}}}}}}
        assert get_field(row, "a.b.c.d.e.f") == "deep_value"

    def test_get_nonexistent_top_level_field(self):
        """Test getting nonexistent top-level field returns None."""
        row = {"existing": "value"}
        assert get_field(row, "nonexistent") is None

    def test_get_nonexistent_nested_field(self):
        """Test getting nonexistent nested field returns None."""
        row = {"metadata": {"language": {"score": 0.95}}}
        assert get_field(row, "metadata.language.confidence") is None
        assert get_field(row, "metadata.nonexistent.field") is None
        assert get_field(row, "nonexistent.nested.field") is None

    def test_get_field_intermediate_non_dict(self):
        """Test getting field through non-dict intermediate returns None."""
        row = {"metadata": "not_a_dict"}
        assert get_field(row, "metadata.language.score") is None

    def test_get_field_intermediate_number(self):
        """Test getting field through numeric intermediate returns None."""
        row = {"metadata": {"score": 42}}
        assert get_field(row, "metadata.score.subscore") is None

    def test_get_field_intermediate_list(self):
        """Test getting field through list intermediate returns None."""
        row = {"metadata": {"items": [1, 2, 3]}}
        assert get_field(row, "metadata.items.length") is None

    def test_get_field_empty_string(self):
        """Test that empty field string returns None."""
        row = {"key": "value"}
        assert get_field(row, "") is None

    def test_get_field_from_empty_dict(self):
        """Test getting field from empty dictionary returns None."""
        row = {}
        assert get_field(row, "any.field") is None


class TestIntegration:
    """Integration tests for add_field and get_field working together."""

    def test_add_then_get_simple(self):
        """Test adding then getting a simple field."""
        row = {}
        set_field(row, "key", "value")
        assert get_field(row, "key") == "value"

    def test_add_then_get_nested(self):
        """Test adding then getting nested fields."""
        row = {}
        set_field(row, "metadata.language.score", 0.95)
        set_field(row, "metadata.language.confidence", 0.87)
        set_field(row, "metadata.processing.timestamp", "2025-06-19")

        assert get_field(row, "metadata.language.score") == 0.95
        assert get_field(row, "metadata.language.confidence") == 0.87
        assert get_field(row, "metadata.processing.timestamp") == "2025-06-19"

    def test_round_trip_various_types(self):
        """Test round trip (add then get) with various value types."""
        row = {}
        test_values = [
            ("string", "test"),
            ("number", 42),
            ("float", 3.14),
            ("boolean", True),
            ("none", None),
            ("list", [1, 2, 3]),
            ("dict", {"nested": "value"}),
        ]

        for field, value in test_values:
            set_field(row, f"data.{field}", value)
            assert get_field(row, f"data.{field}") == value

    def test_modify_existing_structure(self):
        """Test modifying existing structure and retrieving values."""
        row = {"existing": {"field": "original"}}

        # Add to existing structure
        set_field(row, "existing.new_field", "new_value")
        set_field(row, "existing.nested.deep", "deep_value")

        # Verify original and new values
        assert get_field(row, "existing.field") == "original"
        assert get_field(row, "existing.new_field") == "new_value"
        assert get_field(row, "existing.nested.deep") == "deep_value"

    @pytest.mark.parametrize(
        "field,value",
        [
            ("simple", "value"),
            ("nested.field", 42),
            ("deep.nested.field", [1, 2, 3]),
            ("very.deep.nested.field.here", {"key": "value"}),
        ],
    )
    def test_parametrized_round_trip(self, field, value):
        """Parametrized test for round trip operations."""
        row = {}
        set_field(row, field, value)
        assert get_field(row, field) == value


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_single_character_keys(self):
        """Test with single character keys."""
        row = {}
        set_field(row, "a.b.c", "value")
        assert get_field(row, "a.b.c") == "value"

    def test_numeric_string_keys(self):
        """Test with numeric string keys."""
        row = {}
        set_field(row, "0.1.2", "value")
        assert get_field(row, "0.1.2") == "value"

    def test_special_character_keys(self):
        """Test with special characters in keys (not dots)."""
        row = {}
        set_field(row, "key_with_underscore.key-with-dash", "value")
        assert get_field(row, "key_with_underscore.key-with-dash") == "value"

    def test_overwrite_with_different_type(self):
        """Test overwriting field with different type."""
        row = {}
        set_field(row, "field", "string")
        set_field(row, "field", 42)
        assert get_field(row, "field") == 42

        set_field(row, "field", {"nested": "dict"})
        assert get_field(row, "field") == {"nested": "dict"}

    def test_none_value_handling(self):
        """Test handling None values properly."""
        row = {}
        set_field(row, "null_field", None)
        assert get_field(row, "null_field") is None

        # Ensure None is different from missing field
        assert "null_field" in row
        assert get_field(row, "missing_field") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
