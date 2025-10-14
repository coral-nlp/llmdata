from pydantic import Field

from llmdata.core.ops import FilterFn, Row
from llmdata.core.registry import components
from llmdata.core.utils import get_field


@components.add("filter", "token_count")
class TokenCountFilter(FilterFn):
    """Filter text based on token count.

    This filter keeps or removes content based on the number of tokens,
    allowing specification of minimum and maximum token count thresholds.
    """

    # Override base fields with specific _defaults
    name: str = Field(default="token_count_filter", description="Name of the token count filter")
    on: str = Field(default="metadata.token_count", description="Column containing token count")
    if_missing: bool = Field(default=False, description="Return value when token count data is missing")

    # Token filtering specific fields
    min_tokens: int = Field(default=10, description="Minimum number of tokens required", ge=0)
    max_tokens: int | None = Field(
        default=None, description="Maximum number of tokens allowed (None for no limit)", gt=0
    )

    def __call__(self, row: Row) -> bool:
        """Check if the token count is within the specified range."""
        token_count = get_field(row, self.on)
        if not token_count:
            return self.if_missing
        return not (token_count < self.min_tokens or (self.max_tokens and token_count > self.max_tokens))
