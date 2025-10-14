"""Tests for the LLMData writers module."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from llmdata.core.config import get_default_ray_config
from llmdata.core.writers import CSVWriter, JSONLWriter, ParquetWriter, Writer


class TestWriterBaseClass:
    """Test the base Writer class."""

    def test_writer_initialization(self):
        """Test Writer initialization with config."""
        config = get_default_ray_config()
        writer = Writer(config)

        assert writer.config == config
        assert writer.filesystem is None
        assert writer.params == {}

    def test_writer_initialization_with_params(self):
        """Test Writer initialization with parameters."""
        config = get_default_ray_config()
        filesystem = Mock()
        params = {"compression": "snappy"}

        writer = Writer(config, filesystem, **params)

        assert writer.config == config
        assert writer.filesystem == filesystem
        assert writer.params == params

    def test_writer_default_config(self):
        """Test Writer with default config when none provided."""
        writer = Writer(None)

        assert writer.config is not None
        assert hasattr(writer.config, "get_write_kwargs")

    def test_writer_call_not_implemented(self):
        """Test that base Writer call raises NotImplementedError."""
        writer = Writer(get_default_ray_config())
        mock_dataset = Mock()

        with pytest.raises(NotImplementedError):
            writer(mock_dataset, "test_path")


class TestParquetWriter:
    """Test the ParquetWriter class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = get_default_ray_config()
        self.writer = ParquetWriter(self.config)
        self.mock_dataset = Mock()

    def test_parquet_writer_basic_call(self):
        """Test basic ParquetWriter call."""
        self.writer(self.mock_dataset, "test.parquet")

        expected_kwargs = self.config.get_write_kwargs()
        expected_kwargs["compression"] = "snappy"  # Default compression

        self.mock_dataset.write_parquet.assert_called_once_with("test.parquet", filesystem=None, **expected_kwargs)

    def test_parquet_writer_with_compression(self):
        """Test ParquetWriter with custom compression."""
        writer = ParquetWriter(self.config, compression="gzip")
        writer(self.mock_dataset, "test.parquet")

        expected_kwargs = self.config.get_write_kwargs()
        expected_kwargs["compression"] = "gzip"

        self.mock_dataset.write_parquet.assert_called_once_with("test.parquet", filesystem=None, **expected_kwargs)

    def test_parquet_writer_with_multiple_params(self):
        """Test ParquetWriter with multiple parameters."""
        params = {
            "compression": "lz4",
            "row_group_size": 50000,
            "partition_cols": ["year", "month"],
        }
        writer = ParquetWriter(self.config, **params)
        writer(self.mock_dataset, "test.parquet")

        expected_kwargs = self.config.get_write_kwargs()
        expected_kwargs.update(params)

        self.mock_dataset.write_parquet.assert_called_once_with("test.parquet", filesystem=None, **expected_kwargs)

    def test_parquet_writer_with_filesystem(self):
        """Test ParquetWriter with custom filesystem."""
        mock_filesystem = Mock()
        writer = ParquetWriter(self.config, mock_filesystem)
        writer(self.mock_dataset, "s3://bucket/test.parquet")

        expected_kwargs = self.config.get_write_kwargs()
        expected_kwargs["compression"] = "snappy"

        self.mock_dataset.write_parquet.assert_called_once_with(
            "s3://bucket/test.parquet", filesystem=mock_filesystem, **expected_kwargs
        )


class TestJSONLWriter:
    """Test the JSONLWriter class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = get_default_ray_config()
        self.writer = JSONLWriter(self.config)
        self.mock_dataset = Mock()

    def test_jsonl_writer_basic_call(self):
        """Test basic JSONLWriter call."""
        self.writer(self.mock_dataset, "test.jsonl")

        expected_kwargs = self.config.get_write_kwargs()

        self.mock_dataset.write_json.assert_called_once_with("test.jsonl", filesystem=None, **expected_kwargs)

    def test_jsonl_writer_with_params(self):
        """Test JSONLWriter with JSONL-specific parameters."""
        params = {"lines_delimiter": "\n", "force_ascii": True}
        writer = JSONLWriter(self.config, **params)
        writer(self.mock_dataset, "test.jsonl")

        expected_kwargs = self.config.get_write_kwargs()
        expected_kwargs.update(params)

        self.mock_dataset.write_json.assert_called_once_with("test.jsonl", filesystem=None, **expected_kwargs)


class TestCSVWriter:
    """Test the CSVWriter class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = get_default_ray_config()
        self.writer = CSVWriter(self.config)
        self.mock_dataset = Mock()

    def test_csv_writer_basic_call(self):
        """Test basic CSVWriter call."""
        self.writer(self.mock_dataset, "test.csv")

        expected_kwargs = self.config.get_write_kwargs()
        expected_kwargs["include_header"] = True  # Default header

        self.mock_dataset.write_csv.assert_called_once_with("test.csv", filesystem=None, **expected_kwargs)

    def test_csv_writer_with_params(self):
        """Test CSVWriter with CSV-specific parameters."""
        params = {
            "delimiter": ";",
            "header": False,
            "include_header": False,
            "escape_char": "\\",
            "quote_char": '"',
        }
        writer = CSVWriter(self.config, **params)
        writer(self.mock_dataset, "test.csv")

        expected_kwargs = self.config.get_write_kwargs()
        expected_kwargs.update(params)

        self.mock_dataset.write_csv.assert_called_once_with("test.csv", filesystem=None, **expected_kwargs)

    def test_csv_writer_custom_header_override(self):
        """Test that custom include_header parameter overrides default."""
        writer = CSVWriter(self.config, include_header=False)
        writer(self.mock_dataset, "test.csv")

        expected_kwargs = self.config.get_write_kwargs()
        expected_kwargs["include_header"] = False

        self.mock_dataset.write_csv.assert_called_once_with("test.csv", filesystem=None, **expected_kwargs)


class TestWriterRegistry:
    """Test writer registration functionality."""

    def test_writers_are_registered(self):
        """Test that all writers are properly registered."""
        from llmdata.core.registry import components

        expected_writers = ["parquet", "jsonl", "csv"]

        for writer_type in expected_writers:
            assert components.has("writer", writer_type), f"Writer {writer_type} not registered"

    def test_get_writer_classes(self):
        """Test retrieving writer classes from components."""
        from llmdata.core.registry import components

        parquet_writer_cls = components.get("writer", "parquet")
        jsonl_writer_cls = components.get("writer", "jsonl")
        csv_writer_cls = components.get("writer", "csv")

        assert parquet_writer_cls == ParquetWriter
        assert jsonl_writer_cls == JSONLWriter
        assert csv_writer_cls == CSVWriter

    def test_writer_instantiation_from_registry(self):
        """Test instantiating writers from components."""
        from llmdata.core.registry import components

        config = get_default_ray_config()

        for writer_type in ["parquet", "jsonl", "csv"]:
            writer_cls = components.get("writer", writer_type)
            writer = writer_cls(config)

            assert isinstance(writer, Writer)
            assert writer.config == config


class TestWriterIntegration:
    """Integration tests for writers."""

    def test_config_integration(self):
        """Test that writers properly use config kwargs."""
        config = get_default_ray_config(concurrency=5, min_rows_per_file=1000)
        writer = ParquetWriter(config)

        write_kwargs = config.get_write_kwargs()
        assert write_kwargs["concurrency"] == 5
        assert write_kwargs["min_rows_per_file"] == 1000

    def test_filesystem_parameter_passing(self):
        """Test that filesystem parameter is passed correctly."""
        config = get_default_ray_config()
        mock_filesystem = Mock()

        writers = [
            ParquetWriter(config, mock_filesystem),
            JSONLWriter(config, mock_filesystem),
            CSVWriter(config, mock_filesystem),
        ]

        for writer in writers:
            assert writer.filesystem == mock_filesystem

    def test_parameter_handling(self):
        """Test that writer-specific parameters are handled correctly."""
        config = get_default_ray_config()

        # Test with various parameter combinations
        params = {"test_param": "test_value", "another_param": 123}
        writer = ParquetWriter(config, **params)

        assert writer.params == params
        assert "test_param" in writer.params
        assert writer.params["test_param"] == "test_value"

    def test_default_parameter_behavior(self):
        """Test default parameter behavior for different writers."""
        config = get_default_ray_config()
        mock_dataset = Mock()

        # Test ParquetWriter default compression
        parquet_writer = ParquetWriter(config)
        parquet_writer(mock_dataset, "test.parquet")

        # Check that default compression was added
        call_kwargs = mock_dataset.write_parquet.call_args[1]
        assert call_kwargs["compression"] == "snappy"

        mock_dataset.reset_mock()

        # Test CSVWriter default header
        csv_writer = CSVWriter(config)
        csv_writer(mock_dataset, "test.csv")

        # Check that default header was added
        call_kwargs = mock_dataset.write_csv.call_args[1]
        assert call_kwargs["include_header"] is True


class TestWriterErrorHandling:
    """Test error handling in writers."""

    def test_writer_with_invalid_dataset(self):
        """Test writer behavior with invalid dataset."""
        config = get_default_ray_config()
        writer = ParquetWriter(config)

        # Test with None dataset
        with pytest.raises(AttributeError):
            writer(None, "test.parquet")

    def test_writer_config_kwargs_override(self):
        """Test that writer params override config _defaults."""
        config = get_default_ray_config(concurrency=10)
        mock_dataset = Mock()

        # Writer param should override config
        writer = ParquetWriter(config, compression="gzip")
        writer(mock_dataset, "test.parquet")

        call_kwargs = mock_dataset.write_parquet.call_args[1]
        assert call_kwargs["compression"] == "gzip"
        assert call_kwargs["concurrency"] == 10  # From config
