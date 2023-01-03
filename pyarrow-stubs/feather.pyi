from io import IOBase
from typing import overload

import pandas as pd
from pyarrow._feather import FeatherError as FeatherError
from pyarrow.lib import (
    ChunkedArray,
    Codec as Codec,
    NativeFile,
    Schema,
    Table as Table,
    concat_tables as concat_tables,
    schema as schema,
)
from pyarrow.vendored.version import Version as Version
from typing_extensions import Literal

class FeatherDataset:
    paths: list[str]
    validate_schema: bool
    schema: Schema
    def __init__(
        self, path_or_paths: list[str], validate_schema: bool = ...
    ) -> None: ...
    def read_table(self, columns: list[str] | None = ...) -> Table: ...
    def validate_schemas(self, piece: str, table: Table) -> None: ...
    def read_pandas(
        self, columns: list[str] | None = ..., use_threads: bool = ...
    ) -> pd.DataFrame: ...

def check_chunked_overflow(name: str, col: ChunkedArray) -> None: ...
def write_feather(
    df: pd.DataFrame,
    dest: str,
    compression: Literal["zstd", "lz4", "uncompressed"] | None = ...,
    compression_level: int | None = ...,
    chunksize: int | None = ...,
    version: int = ...,
) -> None: ...
@overload
def read_feather(
    source: str,
    columns: list[str] | None = ...,
    use_threads: bool = ...,
    memory_map: Literal[True] = ...,
) -> pd.DataFrame: ...
@overload
def read_feather(
    source: str | NativeFile | IOBase,
    columns: list[str] | None = ...,
    use_threads: bool = ...,
) -> pd.DataFrame: ...
@overload
def read_table(
    source: str | NativeFile | IOBase,
    columns: list[str] | None = ...,
    use_threads: bool = ...,
) -> Table: ...
@overload
def read_table(
    source: str,
    columns: list[str] | None = ...,
    memory_map: Literal[True] = ...,
    use_threads: bool = ...,
) -> Table: ...
