from pathlib import Path
from typing import IO

from .lib import Buffer, NativeFile, Table, _Weakrefable

class FeatherError(Exception): ...

def write_feather(
    table: Table,
    dest: str | IO | Path | NativeFile,
    compression: str | None = None,
    compression_level: int | None = None,
    chunksize: int | None = None,
    version: int = 2,
): ...

class FeatherReader(_Weakrefable):
    def __init__(
        self,
        source: str | IO | Path | NativeFile | Buffer,
        use_memory_map: bool,
        use_threads: bool,
    ) -> None: ...
    @property
    def version(self) -> str: ...
    def read(self) -> Table: ...
    def read_indices(self, indices: list[int]) -> Table: ...
    def read_names(self, names: list[str]) -> Table: ...
