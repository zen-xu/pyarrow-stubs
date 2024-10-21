from typing import IO

from _typeshed import StrPath

from .lib import Buffer, NativeFile, Table, _Weakrefable

class FeatherError(Exception): ...

def write_feather(
    table: Table,
    dest: StrPath | IO | NativeFile,
    compression: str | None = None,
    compression_level: int | None = None,
    chunksize: int | None = None,
    version: int = 2,
): ...

class FeatherReader(_Weakrefable):
    def __init__(
        self,
        source: StrPath | IO | NativeFile | Buffer,
        use_memory_map: bool,
        use_threads: bool,
    ) -> None: ...
    @property
    def version(self) -> str: ...
    def read(self) -> Table: ...
    def read_indices(self, indices: list[int]) -> Table: ...
    def read_names(self, names: list[str]) -> Table: ...
