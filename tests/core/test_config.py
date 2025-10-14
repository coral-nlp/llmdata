import pytest


class TestConfigCreation:
    """Test configuration creation and validation."""

    def test_basic_config_creation(self):
        """Test basic configuration creation."""
        from llmdata.core.config import PipelineConfig

        # Test creating a basic pipeline_config
        config = PipelineConfig(name="test_pipeline", description="Test pipeline")

        assert config.name == "test_pipeline"
        assert config.description == "Test pipeline"

    def test_processor_config_creation(self):
        """Test processor configuration creation."""
        try:
            from core.config import ProcessorConfig

            proc_config = ProcessorConfig(name="test_processor", category="filter", type="language")

            assert proc_config.name == "test_processor"
            assert proc_config.type == "language"

        except ImportError:
            pytest.skip("ProcessorConfig not available")

    def test_connector_config_creation(self):
        """Test connector configuration creation."""
        try:
            from llmdata.core.config import ConnectorConfig

            conn_config = ConnectorConfig(path="dataset.jsonl.gz", format="jsonl")

            assert conn_config.path == "dataset.jsonl.gz"
            assert conn_config.format == "jsonl"

        except ImportError:
            pytest.skip("ConnectorConfig not available")
