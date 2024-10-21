from typing import IO, Literal

from _typeshed import StrPath

from .lib import MemoryPool, Schema, Table, _Weakrefable

class ReadOptions(_Weakrefable):
    use_threads: bool
    block_size: int
    def __init__(self, use_threads: bool | None = None, block_size: int | None = None): ...
    def equals(self, other: ReadOptions) -> bool: ...

class ParseOptions(_Weakrefable):
    explicit_schema: Schema
    newlines_in_values: bool
    unexpected_field_behavior: Literal["ignore", "error", "infer"]
    def __init__(
        self,
        explicit_schema: Schema | None = None,
        newlines_in_values: bool | None = None,
        unexpected_field_behavior: Literal["ignore", "error", "infer"] = "infer",
    ): ...
    def equals(self, other: ParseOptions) -> bool: ...

def read_json(
    input_file: StrPath | IO,
    read_options: ReadOptions | None = None,
    parse_options: ParseOptions | None = None,
    memory_pool: MemoryPool | None = None,
) -> Table: ...
