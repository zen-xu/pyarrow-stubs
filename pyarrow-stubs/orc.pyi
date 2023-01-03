from io import IOBase

from pyarrow._orc import (
    ORCReader as _ORCReader,
    ORCWriter as _ORCWriter,
)
from pyarrow.lib import (
    KeyValueMetadata,
    NativeFile,
    RecordBatch,
    Schema,
    Table,
)

from ._fs import FileSystem

class ORCFile:
    reader: _ORCReader
    def __init__(self, source: str | NativeFile | IOBase) -> None: ...
    @property
    def metadata(self) -> KeyValueMetadata: ...
    @property
    def schema(self) -> Schema: ...
    @property
    def nrows(self) -> int: ...
    @property
    def nstripes(self) -> int: ...
    @property
    def file_version(self) -> str: ...
    @property
    def software_version(self) -> str: ...
    @property
    def compression(self) -> str: ...
    @property
    def compression_size(self) -> int: ...
    @property
    def writer(self) -> str | int: ...
    @property
    def writer_version(self) -> str: ...
    @property
    def row_index_stride(self) -> int: ...
    @property
    def nstripe_statistics(self) -> int: ...
    @property
    def content_length(self) -> int: ...
    @property
    def stripe_statistics_length(self) -> int: ...
    @property
    def file_footer_length(self) -> int: ...
    @property
    def file_postscript_length(self) -> int: ...
    @property
    def file_length(self) -> int: ...
    def read_stripe(self, n: int, columns: list[str] | None = ...) -> RecordBatch: ...
    def read(self, columns: list[str] | None = ...) -> Table: ...

class ORCWriter:
    __doc__: str
    is_open: bool
    writer: _ORCWriter
    def __init__(
        self,
        where: str | NativeFile | IOBase,
        *,
        file_version: str = ...,
        batch_size: int = ...,
        stripe_size: int = ...,
        compression: str = ...,
        compression_block_size: int = ...,
        compression_strategy: str = ...,
        row_index_stride: int = ...,
        padding_tolerance: float = ...,
        dictionary_key_size_threshold: float = ...,
        bloom_filter_columns: list[str] | None = ...,
        bloom_filter_fpp: float = ...,
    ) -> None: ...
    def __del__(self) -> None: ...
    def __enter__(self) -> ORCWriter: ...
    def __exit__(self, *args, **kwargs) -> None: ...
    def write(self, table: Table) -> None: ...
    def close(self) -> None: ...

def read_table(
    source: str | NativeFile | IOBase,
    columns: list[str] | None = ...,
    filesystem: str | FileSystem | None = ...,
) -> Table: ...
def write_table(
    table: Table,
    where: str | NativeFile | IOBase,
    *,
    file_version: str = ...,
    batch_size: int = ...,
    stripe_size: int = ...,
    compression: str = ...,
    compression_block_size: int = ...,
    compression_strategy: str = ...,
    row_index_stride: int = ...,
    padding_tolerance: float = ...,
    dictionary_key_size_threshold: float = ...,
    bloom_filter_columns: list[str] | None = ...,
    bloom_filter_fpp: float = ...,
) -> None: ...
