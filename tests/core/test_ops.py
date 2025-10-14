"""Tests for the LLMData ops module."""

from unittest.mock import Mock

import pytest

from llmdata.core.ops import FilterFn, MapFn, ReduceFn


class TestMapFn:
    """Test the MapFn abstract base class."""

    def test_mapfn_initialization(self):
        """Test MapFn initialization with name and fields."""

        class TestMapFn(MapFn):
            def __call__(self, row):
                return row

        map_fn = TestMapFn("test_map", "field1", "field2")

        assert map_fn.name == "test_map"
        assert map_fn.on == ("field1", "field2")

    def test_mapfn_initialization_no_fields(self):
        """Test MapFn initialization with no fields."""

        class TestMapFn(MapFn):
            def __call__(self, row):
                return row

        map_fn = TestMapFn("test_map")

        assert map_fn.name == "test_map"
        assert map_fn.on == ()

    def test_mapfn_abstract_call(self):
        """Test that MapFn call is abstract."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class MapFn"):
            MapFn("test")

    def test_mapfn_concrete_implementation(self):
        """Test concrete MapFn implementation."""

        class AddFieldMapFn(MapFn):
            def __call__(self, row):
                row["new_field"] = "added_value"
                return row

        map_fn = AddFieldMapFn("add_field")
        row = {"existing": "value"}
        result = map_fn(row)

        assert result == {"existing": "value", "new_field": "added_value"}
        assert map_fn.name == "add_field"


class TestFilterFn:
    """Test the FilterFn abstract base class."""

    def test_filterfn_initialization(self):
        """Test FilterFn initialization with default parameters."""

        class TestFilterFn(FilterFn):
            def __call__(self, row):
                return True

        filter_fn = TestFilterFn("test_filter", "field1", "field2")

        assert filter_fn.name == "test_filter"
        assert filter_fn.on == ("field1", "field2")
        assert filter_fn.if_missing is True

    def test_filterfn_initialization_with_if_missing(self):
        """Test FilterFn initialization with custom if_missing."""

        class TestFilterFn(FilterFn):
            def __call__(self, row):
                return True

        filter_fn = TestFilterFn("test_filter", "field1", if_missing=False)

        assert filter_fn.name == "test_filter"
        assert filter_fn.on == ("field1",)
        assert filter_fn.if_missing is False

    def test_filterfn_initialization_no_fields(self):
        """Test FilterFn initialization with no fields."""

        class TestFilterFn(FilterFn):
            def __call__(self, row):
                return True

        filter_fn = TestFilterFn("test_filter")

        assert filter_fn.name == "test_filter"
        assert filter_fn.on == []
        assert filter_fn.if_missing is True

    def test_filterfn_abstract_call(self):
        """Test that FilterFn call is abstract."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class FilterFn"):
            FilterFn("test")

    def test_filterfn_concrete_implementation(self):
        """Test concrete FilterFn implementation."""

        class LengthFilterFn(FilterFn):
            def __call__(self, row):
                text = row.get("text", "")
                return len(text) > 10

        filter_fn = LengthFilterFn("length_filter", "text")

        assert filter_fn({"text": "short"}) is False
        assert filter_fn({"text": "this is a long enough text"}) is True
        assert filter_fn.name == "length_filter"


class TestReduceFn:
    """Test the ReduceFn abstract base class. Only tests for inheritance, since nothing is actually implemented here."""

    def test_reducefn_is_abstract(self):
        """Test that ReduceFn is abstract and extends AggregateFnV2."""
        from ray.data.aggregate import AggregateFnV2

        # Should not be able to instantiate directly
        with pytest.raises(TypeError):
            ReduceFn()

        # Should be a subclass of AggregateFnV2
        assert issubclass(ReduceFn, AggregateFnV2)


class TestFunctionSignatures:
    """Test function signatures and parameter handling."""

    def test_mapfn_signature_validation(self):
        """Test MapFn signature requirements."""

        class ValidMapFn(MapFn):
            def __call__(self, row):
                return row

        # Should be able to instantiate
        map_fn = ValidMapFn("test")
        assert callable(map_fn)

        # Test call with proper signature
        result = map_fn({"key": "value"})
        assert isinstance(result, dict)

    def test_filterfn_signature_validation(self):
        """Test FilterFn signature requirements."""

        class ValidFilterFn(FilterFn):
            def __call__(self, row):
                return True

        # Should be able to instantiate
        filter_fn = ValidFilterFn("test")
        assert callable(filter_fn)

        # Test call with proper signature
        result = filter_fn({"key": "value"})
        assert isinstance(result, bool)

    def test_field_access_patterns(self):
        """Test common field access patterns in functions."""

        class FieldAccessMap(MapFn):
            def __call__(self, row):
                result = row.copy()
                # Access fields specified in self.on
                for field in self.on:
                    if field in row:
                        result[f"{field}_processed"] = f"processed_{row[field]}"
                return result

        class FieldAccessFilter(FilterFn):
            def __call__(self, row):
                # Check fields specified in self.on
                for field in self.on:
                    if field not in row:
                        return self.if_missing
                    if not row[field]:
                        return False
                return True

        # Test map with field access
        map_fn = FieldAccessMap("field_map", "text", "title")
        row = {"text": "content", "title": "heading", "other": "data"}
        result = map_fn(row)

        assert result["text_processed"] == "processed_content"
        assert result["title_processed"] == "processed_heading"
        assert "other_processed" not in result

        # Test filter with field access
        filter_fn = FieldAccessFilter("field_filter", "text", "title", if_missing=False)

        assert filter_fn({"text": "content", "title": "heading"}) is True
        assert filter_fn({"text": "content"}) is False  # missing title
        assert filter_fn({"text": "", "title": "heading"}) is False  # empty text
