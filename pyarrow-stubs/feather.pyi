from typing import IO, Literal

import pandas as pd

from _typeshed import StrPath
from pyarrow._feather import FeatherError
from pyarrow.lib import Table

__all__ = [
    "FeatherError",
    "FeatherDataset",
    "check_chunked_overflow",
    "write_feather",
    "read_feather",
    "read_table",
]

class FeatherDataset:
    path_or_paths: str | list[str]
    validate_schema: bool

    def __init__(self, path_or_paths: str | list[str], validate_schema: bool = True) -> None: ...
    def read_table(self, columns: list[str] | None = None) -> Table: ...
    def validate_schemas(self, piece, table: Table) -> None: ...
    def read_pandas(
        self, columns: list[str] | None = None, use_threads: bool = True
    ) -> pd.DataFrame: ...

def check_chunked_overflow(name: str, col) -> None: ...
def write_feather(
    df: pd.DataFrame | Table,
    dest: StrPath | IO,
    compression: Literal["zstd", "lz4", "uncompressed"] | None = None,
    compression_level: int | None = None,
    chunksize: int | None = None,
    version: Literal[1, 2] = 2,
) -> None: ...
def read_feather(
    source: StrPath | IO,
    columns: list[str] | None = None,
    use_threads: bool = True,
    memory_map: bool = False,
    **kwargs,
) -> pd.DataFrame: ...
def read_table(
    source: StrPath | IO,
    columns: list[str] | None = None,
    memory_map: bool = False,
    use_threads: bool = True,
) -> Table: ...
