from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field
from ray.data.aggregate import AbsMax, Count, Max, Mean, Min, Quantile, Std, Sum, Unique

from llmdata.core.registry import components

if TYPE_CHECKING:
    from ray.data.aggregate import AggregateFnV2

from llmdata.core.ops import ReduceFn

from .counter import CounterAggregation


class RayAggregation(ReduceFn):
    """Wrapper class around existing ray aggregators."""

    op: Literal["sum", "count", "quantile", "mean", "min", "max", "absmax", "unique", "std"] = Field(
        default="sum", description="Aggregation operation to perform"
    )
    ignore_nulls: bool = Field(default=False, description="Whether to ignore null values in aggregation")
    op_kwargs: dict = Field(
        default_factory=dict,
        description="Additional keyword arguments for the aggregation operation",
    )

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self._op_class = {  # type: ignore[assignment]
            "sum": Sum,
            "count": Count,
            "quantile": Quantile,
            "std": Std,
            "mean": Mean,
            "min": Min,
            "max": Max,
            "absmax": AbsMax,
            "unique": Unique,
        }[self.op]

    def __call__(self) -> "AggregateFnV2":
        """Return the specified native ray aggregation function."""
        return self._op_class(on=self.on, alias_name=self.name, ignore_nulls=self.ignore_nulls, **self.op_kwargs)  # type: ignore[no-any-return]


@components.add("aggregation", "sum")
class SumAggregation(RayAggregation):
    """Sum aggregation over given column."""

    op: Literal["sum"] = Field(default="sum", description="Sum aggregation operation")


@components.add("aggregation", "count")
class CountAggregation(RayAggregation):
    """Count aggregation over given column."""

    op: Literal["count"] = Field(default="count", description="Count aggregation operation")


@components.add("aggregation", "quantile")
class QuantileAggregation(RayAggregation):
    """Quantile aggregation over given column with given q."""

    op: Literal["quantile"] = Field(default="quantile", description="Quantile aggregation operation")
    q: float = Field(default=0.5, description="Quantile value (0.0 to 1.0)", ge=0.0, le=1.0)

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self.op_kwargs = {"q": self.q}


@components.add("aggregation", "mean")
class MeanAggregation(RayAggregation):
    """Mean aggregation over given column."""

    op: Literal["mean"] = Field(default="mean", description="Mean aggregation operation")


@components.add("aggregation", "min")
class MinAggregation(RayAggregation):
    """Min aggregation over given column."""

    op: Literal["min"] = Field(default="min", description="Minimum aggregation operation")


@components.add("aggregation", "max")
class MaxAggregation(RayAggregation):
    """Max aggregation over given column."""

    op: Literal["max"] = Field(default="max", description="Maximum aggregation operation")


@components.add("aggregation", "std")
class StdAggregation(RayAggregation):
    """Str aggregation over given column."""

    op: Literal["std"] = Field(default="std", description="Standard deviation aggregation operation")


@components.add("aggregation", "absmax")
class AbsMaxAggregation(RayAggregation):
    """AbsMax aggregation over given column."""

    op: Literal["absmax"] = Field(default="absmax", description="Absolute maximum aggregation operation")


@components.add("aggregation", "unique")
class UniqueAggregation(RayAggregation):
    """Unique aggregation over given column."""

    op: Literal["unique"] = Field(default="unique", description="Unique values aggregation operation")
