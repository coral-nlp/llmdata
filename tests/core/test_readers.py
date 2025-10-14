"""Tests for the LLMData readers module."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from llmdata.core.config import get_default_ray_config
from llmdata.core.readers import CSVReader, JSONLReader, ParquetReader, Reader, TextReader


class TestReaderBaseClass:
    """Test the base Reader class."""

    def test_reader_initialization(self):
        """Test Reader initialization with config."""
        config = get_default_ray_config()
        reader = Reader(config)

        assert reader.config == config
        assert reader.filesystem is None
        assert reader.params == {}

    def test_reader_initialization_with_params(self):
        """Test Reader initialization with parameters."""
        config = get_default_ray_config()
        filesystem = Mock()
        params = {"columns": ["col1", "col2"]}

        reader = Reader(config, filesystem, **params)

        assert reader.config == config
        assert reader.filesystem == filesystem
        assert reader.params == params

    def test_reader_call_not_implemented(self):
        """Test that base Reader call raises NotImplementedError."""
        reader = Reader(get_default_ray_config())

        with pytest.raises(NotImplementedError):
            reader("test_path")


class TestParquetReader:
    """Test the ParquetReader class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = get_default_ray_config()
        self.reader = ParquetReader(self.config)

    @patch("ray.data.read_parquet")
    def test_parquet_reader_basic_call(self, mock_read_parquet):
        """Test basic ParquetReader call."""
        mock_dataset = Mock()
        mock_read_parquet.return_value = mock_dataset

        result = self.reader("test.parquet")

        mock_read_parquet.assert_called_once_with("test.parquet", filesystem=None, **self.config.get_read_kwargs())
        assert result == mock_dataset

    @patch("ray.data.read_parquet")
    def test_parquet_reader_with_columns(self, mock_read_parquet):
        """Test ParquetReader with column selection."""
        mock_dataset = Mock()
        mock_read_parquet.return_value = mock_dataset

        reader = ParquetReader(self.config, columns=["col1", "col2"])
        result = reader("test.parquet")

        expected_kwargs = self.config.get_read_kwargs()
        expected_kwargs["columns"] = ["col1", "col2"]

        mock_read_parquet.assert_called_once_with("test.parquet", filesystem=None, **expected_kwargs)

    @patch("ray.data.read_parquet")
    def test_parquet_reader_with_filesystem(self, mock_read_parquet):
        """Test ParquetReader with custom filesystem."""
        mock_dataset = Mock()
        mock_read_parquet.return_value = mock_dataset
        mock_filesystem = Mock()

        reader = ParquetReader(self.config, mock_filesystem)
        result = reader("s3://bucket/test.parquet")

        mock_read_parquet.assert_called_once_with(
            "s3://bucket/test.parquet", filesystem=mock_filesystem, **self.config.get_read_kwargs()
        )


class TestJSONLReader:
    """Test the JSONLReader class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = get_default_ray_config()
        self.reader = JSONLReader(self.config)

    @patch("ray.data.read_json")
    def test_jsonl_reader_basic_call(self, mock_read_json):
        """Test basic JSONLReader call."""
        mock_dataset = Mock()
        mock_read_json.return_value = mock_dataset

        result = self.reader("test.jsonl")

        mock_read_json.assert_called_once_with("test.jsonl", filesystem=None, **self.config.get_read_kwargs())
        assert result == mock_dataset


class TestCSVReader:
    """Test the CSVReader class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = get_default_ray_config()
        self.reader = CSVReader(self.config)

    @patch("ray.data.read_csv")
    def test_csv_reader_basic_call(self, mock_read_csv):
        """Test basic CSVReader call."""
        mock_dataset = Mock()
        mock_read_csv.return_value = mock_dataset

        result = self.reader("test.csv")

        mock_read_csv.assert_called_once_with("test.csv", filesystem=None, **self.config.get_read_kwargs())
        assert result == mock_dataset

    @patch("ray.data.read_csv")
    def test_csv_reader_with_params(self, mock_read_csv):
        """Test CSVReader with CSV-specific parameters."""
        mock_dataset = Mock()
        mock_read_csv.return_value = mock_dataset

        params = {
            "delimiter": ",",
            "header": "infer",
            "names": ["col1", "col2"],
            "dtype": {"col1": "str"},
            "skiprows": 1,
        }
        reader = CSVReader(self.config, **params)
        result = reader("test.csv")

        expected_kwargs = self.config.get_read_kwargs()
        expected_kwargs.update(params)

        mock_read_csv.assert_called_once_with("test.csv", filesystem=None, **expected_kwargs)


class TestTextReader:
    """Test the TextReader class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = get_default_ray_config()
        self.reader = TextReader(self.config)

    @patch("ray.data.read_text")
    def test_text_reader_basic_call(self, mock_read_text):
        """Test basic TextReader call."""
        mock_dataset = Mock()
        mock_read_text.return_value = mock_dataset

        result = self.reader("test.txt")

        mock_read_text.assert_called_once_with("test.txt", filesystem=None, **self.config.get_read_kwargs())
        assert result == mock_dataset

    @patch("ray.data.read_text")
    def test_text_reader_with_encoding(self, mock_read_text):
        """Test TextReader with encoding parameter."""
        mock_dataset = Mock()
        mock_read_text.return_value = mock_dataset

        reader = TextReader(self.config, encoding="utf-8")
        result = reader("test.txt")

        expected_kwargs = self.config.get_read_kwargs()
        expected_kwargs["encoding"] = "utf-8"

        mock_read_text.assert_called_once_with("test.txt", filesystem=None, **expected_kwargs)


class TestReaderRegistry:
    """Test reader registration functionality."""

    def test_readers_are_registered(self):
        """Test that all readers are properly registered."""
        from llmdata.core.registry import components

        expected_readers = ["parquet", "jsonl", "csv", "text"]

        for reader_type in expected_readers:
            assert components.has("reader", reader_type), f"Reader {reader_type} not registered"

    def test_get_reader_classes(self):
        """Test retrieving reader classes from components."""
        from llmdata.core.registry import components

        parquet_reader_cls = components.get("reader", "parquet")
        jsonl_reader_cls = components.get("reader", "jsonl")
        csv_reader_cls = components.get("reader", "csv")
        text_reader_cls = components.get("reader", "text")

        assert parquet_reader_cls == ParquetReader
        assert jsonl_reader_cls == JSONLReader
        assert csv_reader_cls == CSVReader
        assert text_reader_cls == TextReader

    def test_reader_instantiation_from_registry(self):
        """Test instantiating readers from components."""
        from llmdata.core.registry import components

        config = get_default_ray_config()

        for reader_type in ["parquet", "jsonl", "csv", "text"]:
            reader_cls = components.get("reader", reader_type)
            reader = reader_cls(config)

            assert isinstance(reader, Reader)
            assert reader.config == config


class TestReaderIntegration:
    """Integration tests for readers."""

    def test_config_integration(self):
        """Test that readers properly use config kwargs."""
        config = get_default_ray_config(concurrency=5, override_num_blocks=10)
        reader = ParquetReader(config)

        read_kwargs = config.get_read_kwargs()
        assert read_kwargs["concurrency"] == 5
        assert read_kwargs["override_num_blocks"] == 10

    def test_filesystem_parameter_passing(self):
        """Test that filesystem parameter is passed correctly."""
        config = get_default_ray_config()
        mock_filesystem = Mock()

        readers = [
            ParquetReader(config, mock_filesystem),
            JSONLReader(config, mock_filesystem),
            CSVReader(config, mock_filesystem),
            TextReader(config, mock_filesystem),
        ]

        for reader in readers:
            assert reader.filesystem == mock_filesystem

    def test_parameter_handling(self):
        """Test that reader-specific parameters are handled correctly."""
        config = get_default_ray_config()

        # Test with various parameter combinations
        params = {"test_param": "test_value", "another_param": 123}
        reader = ParquetReader(config, **params)

        assert reader.params == params
        assert "test_param" in reader.params
        assert reader.params["test_param"] == "test_value"
