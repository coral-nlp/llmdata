from collections import Counter
from typing import Callable

from pydantic import Field
from ray.data.aggregate import AggregateFnV2
from ray.data.block import AggType, Block, BlockAccessor

from llmdata.core.ops import ReduceFn
from llmdata.core.registry import components


class CounterAggFn(AggregateFnV2):
    """Aggregation function to combine counter objects across a dataset."""

    def __init__(
        self,
        name: str,
        zero_factory: Callable[[], AggType],
        on: str | None,
        ignore_nulls: bool,
        top_k: int | None = None,
    ):
        super().__init__(name, zero_factory, on=on, ignore_nulls=ignore_nulls)
        self.top_k = top_k

    def combine(self, current_accumulator: AggType, new: AggType) -> AggType:
        """Combine a new partial aggregation result with the current accumulator."""
        return Counter(new) + Counter(current_accumulator)

    def aggregate_block(self, block: Block) -> AggType:
        """Aggregate data within a single block."""
        counter: Counter = Counter()
        for c in BlockAccessor.for_block(block).select([self._target_col_name]):
            counter += Counter(c)
        return counter.most_common(self.top_k)

    def finalize(self, accumulator: AggType) -> AggType:
        """Transform the final accumulated state into the desired output."""
        return accumulator.most_common(self.top_k)


@components.add("aggregation", "counter")
class CounterAggregation(ReduceFn):
    """Aggregation over  counters objects."""

    name: str = Field(title="The name of this reduce function.")
    on: str = Field(title="The column to apply this function to.")
    top_k: int | None = Field(
        title="The top K highest counted values to keep during aggregation (reduces memory overhead)."
    )

    def __call__(self) -> "AggregateFnV2":
        """Construct an aggregation function for ray to consume."""
        return CounterAggFn(name=self.name, on=self.on, zero_factory=lambda: Counter(), ignore_nulls=True)
