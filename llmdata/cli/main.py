#!/usr/bin/env python3

import logging
import sys

from jsonargparse import CLI
from jsonargparse.typing import Path_fc, Path_fr

from llmdata.core.config import PipelineConfig, RayConfig
from llmdata.core.pipeline import DataPipeline
from llmdata.core.registry import components


def setup_logging(level: str = "INFO") -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


class MainCLI:
    """Main CLI class."""

    def run(
        self,
        config_path: Path_fr,
        input_path: str | None = None,
        output_path: str | None = None,
        ray: RayConfig | None = None,
        log_level: str = "INFO",
    ) -> None:
        """Run a processing pipeline from configuration.

        Args:
            config_path: Path to pipeline configuration file
            input_path: Override input path from pipeline configuration
            output_path: Override output path from pipeline configuration
            ray: Ray configuration overrides
            log_level: Logging level

        """
        setup_logging(log_level)

        # Load pipeline configuration
        config = PipelineConfig.from_yaml(config_path)
        pipeline = DataPipeline(config)

        # Override paths if provided
        if input_path is not None:
            pipeline.config.input.path = input_path  # type: ignore[union-attr]
        if output_path is not None:
            pipeline.config.output.path = output_path  # type: ignore[union-attr]

        # Override Ray configuration if provided
        if ray:
            ray_dict = ray.__dict__
            ray_overrides = {k: v for k, v in ray_dict.items() if v is not None}
            if ray_overrides:
                for key, value in ray_overrides.items():
                    setattr(pipeline.config.ray_config, key, value)

        # Execute pipeline
        print(f"Running pipeline: {pipeline.config.name}")
        if pipeline.config.description:
            print(f"Description: {pipeline.config.description}")

        results = pipeline.run(
            read_kwargs=config.input.params,
            process_kwargs=config.process_kwargs,
            write_kwargs=config.output.params if config.output is not None else None,
            aggregate_kwargs=config.aggregation_kwargs,
        )

        # Display results
        print("\n=== DataPipeline Results ===")
        print(results)

    def list(self, category: str | None = None) -> None:
        """List available components.

        Args:
            category: Filter by component category

        """
        print("=== Available Components ===")
        categories = [category] if category is not None else components.categories()
        for cat in categories:
            print(f"\n{cat}:")
            for component_name in components.components(cat):
                print(f"  - {component_name}")

    def validate(self, path: Path_fr) -> None:
        """Validate data files or configurations.

        Args:
            path: Path to data file or configuration to validate

        """
        try:
            config = PipelineConfig.from_yaml(path)
            print(f"✓ Configuration is valid: {config.name}")
        except Exception as e:
            print(f"✗ Configuration validation failed: {e}", file=sys.stderr)
            sys.exit(1)

    def export_schemas(self, output_path: Path_fc, category: str | None = None) -> None:
        """Export JSON schemas for all processors and aggregations.

        Args:
            output_path: Path to write JSON schema file
            category: Optional category filter (e.g., 'extractor', 'filter')

        """
        # Import all processors to register them

        try:
            components.export_schemas(output_path, category)
            if category:
                print(f"✓ Exported {category} schemas to {output_path}")
            else:
                print(f"✓ Exported all schemas to {output_path}")
        except Exception as e:
            print(f"✗ Failed to export schemas: {e}", file=sys.stderr)
            sys.exit(1)


def main() -> None:
    """Execute the main CLI entrypoint."""
    CLI(MainCLI)


if __name__ == "__main__":
    main()
