import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import PipelineConfig

from ray.data import DataContext, Dataset

from .filesystem import get_fs
from .ops import FilterFn, MapFn, ReduceFn
from .readers import Reader
from .registry import components
from .utils import get_field
from .writers import Writer


class DataPipeline:
    """Orchestration class for data processing.

    The DataPipeline class coordinates the execution of data processing workflows
    based on a (valid) PipelineConfig.
    """

    def __init__(self, pipeline_config: "PipelineConfig") -> None:
        """Initialize pipeline with validated configuration.

        Args:
            pipeline_config: A PipelineConfig instance containing
                           all necessary configuration for the pipeline execution.

        """
        self.config = pipeline_config
        self.dataset: Dataset | None = None
        self._configure_context()

    def _configure_context(self) -> None:
        """Configure Ray DataContext with settings."""
        context = DataContext.get_current()
        for key, value in self.config.ray_config.get_context_config().items():
            setattr(context, key, value)

    def read(self, **kwargs: Any) -> None:
        """Load data using the configured reader."""
        if not self.config.input:
            raise ValueError("No input specified")

        reader: Reader = components.get("reader", self.config.input.format)(self.config.ray_config, **kwargs)
        self.dataset = reader(path=self.config.input.path)

    def process(self, **kwargs: Any) -> None:
        """Execute all processing steps."""
        if self.dataset is None:
            raise ValueError("No data. Did you call read() first?")

        for processor_config in self.config.processors:
            if not processor_config.enabled:
                continue
            processor = components.get(processor_config.category, processor_config.type)(**processor_config.params)
            if isinstance(processor, FilterFn):
                self.dataset = self.dataset.filter(processor, **self.config.ray_config.get_filter_kwargs())
            elif isinstance(processor, MapFn):
                self.dataset = self.dataset.map(processor, **self.config.ray_config.get_map_kwargs())
            else:
                raise ValueError(
                    f"Processors must either inherit from 'MapFn'  or 'FilterFn'; got {type(processor).__bases__}"
                )

    def write(self, **kwargs: Any) -> None:
        """Save data using the configured writer."""
        if self.dataset is None:
            raise ValueError("No data. Did you call read() or process() first?")

        writer: Writer = components.get("writer", self.config.output.format)(self.config.ray_config, **kwargs)  # type: ignore[union-attr]
        writer(self.dataset, path=self.config.output.path)  # type: ignore[union-attr]

    def aggregate(
        self, output_path: str | None = None, groupby: list[str] | str | None = None, **kwargs: Any
    ) -> dict[str, Any] | None:
        """Execute all aggregation steps."""
        aggs: list[ReduceFn] = []
        # Make locally available
        ds = self.dataset
        for agg_config in self.config.aggregations:  # type: ignore[union-attr]
            # Skip if not enabled
            if not agg_config.enabled:
                continue
            # Instantiate the aggregation
            agg: ReduceFn = components.get("aggregation", agg_config.type)(**agg_config.params)
            aggs.append(agg)
            # Add target column (in case its nested agg can't access it)
            ds = ds.add_column(agg.on, lambda row, field=agg.on: get_field(row, field=field))  # type: ignore[union-attr]

        if groupby is not None:
            if isinstance(groupby, str):
                groupby = [groupby]
            # Add group columns (in case they are nested, groupby can't access it)
            for col in groupby:  # type: ignore
                ds = ds.add_column(col=col, fn=lambda row, field=col: get_field(row, field))  # type: ignore
            res: dict[str, Any] = ds.groupby(groupby).aggregate(aggs).take_all()  # type: ignore
        else:
            res: dict[str, Any] = ds.aggregate(aggs)  # type: ignore

        if not res:
            return None

        if output_path:
            with get_fs(output_path, "fsspec").open(output_path, "w") as f:
                json.dump(res, f, indent=4, sort_keys=True)

        return res

    def run(
        self,
        read_kwargs: dict[str, Any] | None = None,
        process_kwargs: dict[str, Any] | None = None,
        write_kwargs: dict[str, Any] | None = None,
        aggregate_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Execute the complete pipeline."""
        self.read(**(read_kwargs or {}))
        if self.config.processors is not None:
            self.process(**(process_kwargs or {}))
        if self.config.output is not None:
            self.write(**(write_kwargs or {}))
        if self.config.aggregations is not None:
            aggregations = self.aggregate(**(aggregate_kwargs or {}))
            return aggregations
        return None
