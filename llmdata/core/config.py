from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from yamlcore import CoreLoader  # To circumvent 'on' being parsed as true from yaml


class RayConfig(BaseModel):
    """Configuration for Ray execution environment and resource management."""

    model_config = ConfigDict(extra="forbid")

    # Core resource settings
    target_max_block_size_mb: int = Field(default=128, description="Maximum block size in MB", gt=0)
    override_num_blocks: int | None = Field(default=None, description="Override number of blocks", gt=0)
    concurrency: int | None = Field(default=None, description="Max concurrent tasks", gt=0)

    # Memory and CPU allocation
    num_cpus_per_task: float = Field(default=1.0, description="CPU cores per task", gt=0)
    memory_per_task_mb: int | None = Field(default=None, description="Memory per task in MB", gt=0)

    # Batch processing
    batch_size: int | None = Field(default=None, description="Batch size for map_batches operations", gt=0)

    # Output
    min_rows_per_file: int | None = Field(default=None, description="Minimum rows per output file", gt=0)

    def get_context_config(self) -> dict[str, Any]:
        """Get kwargs for Ray DataContext configuration."""
        return {
            "target_max_block_size": self.target_max_block_size_mb * 1024 * 1024,
            "enable_auto_log_stats": False,
            "enable_progress_bars": False,
        }

    def get_read_kwargs(self) -> dict[str, Any]:
        """Get kwargs for Ray read operations."""
        kwargs: dict[str, Any] = {}
        if self.override_num_blocks:
            kwargs["override_num_blocks"] = self.override_num_blocks
        if self.concurrency:
            kwargs["concurrency"] = self.concurrency
        if self.num_cpus_per_task:
            if "ray_remote_args" not in kwargs:
                kwargs["ray_remote_args"] = {}
            kwargs["ray_remote_args"].update({"num_cpus": self.num_cpus_per_task})
        if self.memory_per_task_mb:
            if "ray_remote_args" not in kwargs:
                kwargs["ray_remote_args"] = {}
            kwargs["ray_remote_args"].update({"memory": self.memory_per_task_mb * 1024 * 1024})
        return kwargs

    def get_map_kwargs(self) -> dict[str, int | float | str | None]:
        """Get kwargs for Ray map operations."""
        kwargs: dict[str, int | float | str | None] = {}
        if self.concurrency:
            kwargs["concurrency"] = self.concurrency
        if self.num_cpus_per_task != 1.0:
            kwargs["num_cpus"] = self.num_cpus_per_task
        if self.memory_per_task_mb:
            kwargs["memory"] = self.memory_per_task_mb * 1024 * 1024
        return kwargs

    def get_filter_kwargs(self) -> dict[str, int | float | str | None]:
        """Get kwargs for Ray filter operations."""
        return self.get_map_kwargs()

    def get_batch_kwargs(self) -> dict[str, int | float | str | None]:
        """Get kwargs for Ray map_batches operations."""
        kwargs = self.get_map_kwargs()
        if self.batch_size:
            kwargs["batch_size"] = self.batch_size
        kwargs["batch_format"] = "numpy"
        return kwargs

    def get_write_kwargs(self) -> dict[str, Any]:
        """Get kwargs for Ray write operations."""
        kwargs = {}
        if self.min_rows_per_file:
            kwargs["min_rows_per_file"] = self.min_rows_per_file
        if self.concurrency:
            kwargs["concurrency"] = self.concurrency
        return kwargs


def get_default_ray_config(
    max_block_size_mb: int | None = 128,
    override_num_blocks: int | None = None,
    concurrency: int | None = None,
    num_cpus_per_task: float | None = 1,
    batch_size: int | None = 256,
    min_rows_per_file: int | None = 10_000,
) -> RayConfig:
    """Get default Ray configuration for typical use cases."""
    return RayConfig(
        target_max_block_size_mb=max_block_size_mb,  # type: ignore[arg-type]
        override_num_blocks=override_num_blocks,
        concurrency=concurrency,
        num_cpus_per_task=num_cpus_per_task,
        batch_size=batch_size,
        min_rows_per_file=min_rows_per_file,
    )


class ProcessorConfig(BaseModel):
    """Configuration for a single processor (map or filter op)."""

    model_config = ConfigDict(extra="forbid")

    category: str = Field(description="Category of processor (e.g., 'extractor', 'filter', 'tagger')")
    type: str = Field(description="Type identifier for the specific processor implementation")
    params: dict[str, Any] = Field(default_factory=dict, description="Parameters specific to this processor")
    enabled: bool = Field(default=True, description="Whether this processor is enabled")


class AggregationConfig(BaseModel):
    """Configuration for a single stat module (aggregation op)."""

    model_config = ConfigDict(extra="forbid")

    category: str = Field(description="Category of aggregation")
    type: str = Field(description="Type identifier for the specific aggregation implementation")
    params: dict[str, Any] = Field(default_factory=dict, description="Parameters specific to this aggregation")
    enabled: bool = Field(default=True, description="Whether this aggregation is enabled")


class ConnectorConfig(BaseModel):
    """Configuration for data connectors (readers/writers)."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(description="Path(s) to data source or output destination")
    format: str = Field(description="Data format (e.g., 'parquet', 'jsonl', 'csv')")
    params: dict[str, Any] = Field(default_factory=dict, description="Format-specific parameters")


class PipelineConfig(BaseModel):
    """Complete pipeline configuration."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Name of the pipeline")
    description: str | None = Field(default=None, description="Description of what this pipeline does")

    # Data sources
    input: ConnectorConfig = Field(default_factory=ConnectorConfig, description="Input data source configuration")

    # Processing steps
    processors: list[ProcessorConfig] = Field(default_factory=list, description="List of processing steps")
    process_kwargs: dict[str, Any] = Field(default_factory=dict, description="kwargs to supply to all processors")

    # Data sinks
    output: ConnectorConfig | None = Field(default=None, description="Output destination configuration")

    # Stats to compute after processing
    aggregations: list[AggregationConfig] | None = Field(default=None, description="List of aggregation operations")
    aggregation_kwargs: dict[str, Any] = Field(default_factory=dict, description="kwargs to supply to all aggregators")

    # Ray configuration
    ray_config: RayConfig = Field(default_factory=RayConfig, description="Ray execution configuration")

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "PipelineConfig":
        """Load configuration from YAML file."""
        with open(yaml_path) as f:
            data = yaml.load(f, Loader=CoreLoader)  # nosec
        return cls.model_validate(data)  # type: ignore[no-any-return]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineConfig":
        """Create configuration from dictionary."""
        return cls.model_validate(data)  # type: ignore[no-any-return]

    def save_yaml(self, yaml_path: str | Path) -> None:
        """Save configuration to YAML file."""
        with open(yaml_path, "w") as f:
            # Convert to dict, excluding unset values
            data = self.model_dump(exclude_unset=True, exclude_none=True)
            yaml.dump(data, f, default_flow_style=False)

    @model_validator(mode="after")
    def validate_pipeline(self) -> "PipelineConfig":
        """Validate the complete pipeline configuration."""
        from llmdata.core.registry import components

        # Validate that all processors are registered
        for processor in self.processors:
            if not components.has(processor.category, processor.type):
                raise ValueError(
                    f"Unknown processor: {processor.category}.{processor.type}. "
                    f"Available in {processor.category}: {components.components(processor.category)}"
                )

            # Validate processor configuration against its schema
            try:
                components.validate_config(processor.category, processor.type, processor.params)
            except Exception as e:
                raise ValueError(f"Invalid configuration for {processor.category}.{processor.type}: {e}")

        # Validate aggregations
        if self.aggregations is not None:
            for aggregation in self.aggregations:
                if not components.has(aggregation.category, aggregation.type):
                    raise ValueError(
                        f"Unknown aggregation: {aggregation.category}.{aggregation.type}. "
                        f"Available in {aggregation.category}: {components.components(aggregation.category)}"
                    )

                # Validate aggregation configuration against its schema
                try:
                    components.validate_config(aggregation.category, aggregation.type, aggregation.params)
                except Exception as e:
                    raise ValueError(f"Invalid configuration for {aggregation.category}.{aggregation.type}: {e}")

        return self
