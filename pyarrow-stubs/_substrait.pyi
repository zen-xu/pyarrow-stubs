from typing import (
    Callable,
    NamedTuple,
)

from pyarrow.lib import (
    Buffer,
    RecordBatchReader,
    Table,
)

def _parse_json_plan(plan: bytes) -> Buffer: ...
def get_supported_functions() -> list[str]: ...
def run_query(
    plan: Buffer | bytes, table_provider: Callable[[NamedTuple], Table] | None = ...
) -> RecordBatchReader: ...
