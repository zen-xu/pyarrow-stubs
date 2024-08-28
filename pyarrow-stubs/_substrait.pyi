from typing import Callable

from ._compute import Expression
from .lib import Buffer, RecordBatchReader, Schema, Table, _Weakrefable

def run_query(
    plan: Buffer | int,
    *,
    table_provider: Callable[[list[str], Schema], Table] | None = None,
    use_threads: bool = True,
) -> RecordBatchReader: ...
def _parse_json_plan(plan: bytes) -> Buffer: ...
def serialize_expressions(
    exprs: list[Expression],
    names: list[str],
    schema: Schema,
    *,
    allow_arrow_extensions: bool = False,
) -> Buffer: ...

class BoundExpressions(_Weakrefable):
    @property
    def schema(self) -> Schema: ...
    @property
    def expressions(self) -> dict[str, Expression]: ...

def deserialize_expressions(buf: Buffer | bytes) -> BoundExpressions: ...
def get_supported_functions() -> list[str]: ...
